#include "causal_softmax_bang.h"

#include "../../../devices/bang/common_bang.h"
#include <cnnl_extra.h>

#include <algorithm>
#include <vector>

namespace op::causal_softmax::bang {

static size_t alignUp(size_t value, size_t alignment = 128) {
    return (value + alignment - 1) / alignment * alignment;
}

static infiniStatus_t setExpandedTensor(
    cnnlTensorDescriptor_t desc,
    infiniopTensorDescriptor_t tensor_desc,
    const CausalSoftmaxInfo &info) {

    int dims[4] = {
        static_cast<int>(info.batch_size), 1,
        static_cast<int>(info.seq_len),
        static_cast<int>(info.total_seq_len)};
    int strides[4];
    if (tensor_desc->ndim() == 2) {
        strides[3] = static_cast<int>(tensor_desc->stride(1));
        strides[2] = static_cast<int>(tensor_desc->stride(0));
        strides[1] = strides[2] * dims[2];
        strides[0] = strides[1];
    } else {
        strides[3] = static_cast<int>(tensor_desc->stride(2));
        strides[2] = static_cast<int>(tensor_desc->stride(1));
        strides[1] = strides[2] * dims[2];
        strides[0] = static_cast<int>(tensor_desc->stride(0));
    }
    CHECK_BANG(cnnlSetTensorDescriptorEx(
        desc, CNNL_LAYOUT_ARRAY,
        device::bang::getCnnlDtype(info.dtype), 4, dims, strides));
    return INFINI_STATUS_SUCCESS;
}

struct Descriptor::Opaque {
    std::shared_ptr<device::bang::Handle::Internal> internal;
    cnnlTensorDescriptor_t x_desc = nullptr;
    cnnlTensorDescriptor_t y_desc = nullptr;
    cnnlTensorDescriptor_t contiguous_x_desc = nullptr;
    cnnlTensorDescriptor_t contiguous_y_desc = nullptr;
    cnnlTensorDescriptor_t mask_desc = nullptr;
    void *mask = nullptr;
    size_t x_offset = 0;
    size_t y_offset = 0;
    size_t copy_workspace_offset = 0;
    size_t copy_workspace_size = 0;
    bool pack_x = false;
    bool unpack_y = false;

    ~Opaque() {
        if (x_desc) {
            cnnlDestroyTensorDescriptor(x_desc);
        }
        if (y_desc) {
            cnnlDestroyTensorDescriptor(y_desc);
        }
        if (contiguous_x_desc) {
            cnnlDestroyTensorDescriptor(contiguous_x_desc);
        }
        if (contiguous_y_desc) {
            cnnlDestroyTensorDescriptor(contiguous_y_desc);
        }
        if (mask_desc) {
            cnnlDestroyTensorDescriptor(mask_desc);
        }
        if (mask) {
            cnrtFree(mask);
        }
    }
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {

    auto result = CausalSoftmaxInfo::create(y_desc, x_desc);
    CHECK_RESULT(result);
    auto info = result.take();
    CHECK_OR_RETURN(info.total_seq_len <= 2048, INFINI_STATUS_NOT_IMPLEMENTED);
    CHECK_OR_RETURN(!y_desc->hasBroadcastDim(), INFINI_STATUS_BAD_TENSOR_STRIDES);

    auto handle = reinterpret_cast<device::bang::Handle *>(handle_);
    auto opaque = new Opaque{handle->internal()};
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->x_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->y_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->contiguous_x_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->contiguous_y_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->mask_desc));
    CHECK_STATUS(setExpandedTensor(opaque->x_desc, x_desc, info));
    CHECK_STATUS(setExpandedTensor(opaque->y_desc, y_desc, info));

    int dims[4] = {
        static_cast<int>(info.batch_size), 1,
        static_cast<int>(info.seq_len),
        static_cast<int>(info.total_seq_len)};
    CHECK_BANG(cnnlSetTensorDescriptor(
        opaque->contiguous_x_desc, CNNL_LAYOUT_ARRAY,
        device::bang::getCnnlDtype(info.dtype), 4, dims));
    CHECK_BANG(cnnlSetTensorDescriptor(
        opaque->contiguous_y_desc, CNNL_LAYOUT_ARRAY,
        device::bang::getCnnlDtype(info.dtype), 4, dims));

    int mask_dims[4] = {
        static_cast<int>(info.batch_size), 1, static_cast<int>(info.seq_len),
        static_cast<int>(info.total_seq_len)};
    CHECK_BANG(cnnlSetTensorDescriptor(
        opaque->mask_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_BOOL, 4, mask_dims));
    const size_t mask_matrix_size = info.seq_len * info.total_seq_len;
    std::vector<uint8_t> host_mask(info.batch_size * mask_matrix_size, 0);
    for (size_t batch = 0; batch < info.batch_size; ++batch) {
        for (size_t i = 0; i < info.seq_len; ++i) {
            const size_t first_masked = info.total_seq_len - info.seq_len + i + 1;
            for (size_t j = first_masked; j < info.total_seq_len; ++j) {
                host_mask[batch * mask_matrix_size + i * info.total_seq_len + j] = 1;
            }
        }
    }
    CHECK_INTERNAL(cnrtSetDevice(handle->device_id), cnrtSuccess);
    CHECK_INTERNAL(cnrtMalloc(&opaque->mask, host_mask.size()), cnrtSuccess);
    CHECK_INTERNAL(cnrtMemcpy(
                       opaque->mask, host_mask.data(), host_mask.size(), cnrtMemcpyHostToDev),
                   cnrtSuccess);

    opaque->pack_x = !x_desc->isContiguous();
    opaque->unpack_y = !y_desc->isContiguous();
    CHECK_STATUS(opaque->internal->useCnnl(
        nullptr, [&](cnnlHandle_t cnnl_handle) {
            size_t size = 0;
            if (opaque->pack_x) {
                CHECK_BANG(cnnlGetCopyWorkspaceSize(
                    cnnl_handle, opaque->x_desc, opaque->contiguous_x_desc, &size));
                opaque->copy_workspace_size = std::max(opaque->copy_workspace_size, size);
            }
            if (opaque->unpack_y) {
                CHECK_BANG(cnnlGetCopyWorkspaceSize(
                    cnnl_handle, opaque->contiguous_y_desc, opaque->y_desc, &size));
                opaque->copy_workspace_size = std::max(opaque->copy_workspace_size, size);
            }
            return INFINI_STATUS_SUCCESS;
        }));

    const size_t tensor_bytes = info.batch_size * info.seq_len
                              * info.total_seq_len * infiniSizeOf(info.dtype);
    size_t workspace_size = 0;
    if (opaque->pack_x) {
        opaque->x_offset = workspace_size;
        workspace_size += alignUp(tensor_bytes);
    }
    if (opaque->unpack_y) {
        opaque->y_offset = workspace_size;
        workspace_size += alignUp(tensor_bytes);
    }
    if (opaque->copy_workspace_size > 0) {
        opaque->copy_workspace_offset = workspace_size;
        workspace_size += alignUp(opaque->copy_workspace_size);
    }

    *desc_ptr = new Descriptor(
        opaque, std::move(info), workspace_size,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y, const void *x, void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto base = reinterpret_cast<char *>(workspace);
    const void *softmax_x = _opaque->pack_x ? base + _opaque->x_offset : x;
    void *softmax_y = _opaque->unpack_y ? base + _opaque->y_offset : y;
    auto softmax_x_desc = _opaque->pack_x ? _opaque->contiguous_x_desc : _opaque->x_desc;
    auto softmax_y_desc = _opaque->unpack_y ? _opaque->contiguous_y_desc : _opaque->y_desc;
    void *copy_workspace = _opaque->copy_workspace_size == 0
                             ? nullptr
                             : base + _opaque->copy_workspace_offset;
    auto queue = reinterpret_cast<cnrtQueue_t>(stream);
    CHECK_STATUS(_opaque->internal->useCnnl(
        queue, [&](cnnlHandle_t cnnl_handle) {
            if (_opaque->pack_x) {
                CHECK_BANG(cnnlCopy_v2(
                    cnnl_handle, _opaque->x_desc, x,
                    _opaque->contiguous_x_desc, const_cast<void *>(softmax_x),
                    copy_workspace, _opaque->copy_workspace_size));
            }
            CHECK_BANG(cnnlMaskedSoftmax(
                cnnl_handle, CNNL_MASKED_SOFTMAX_MASKED_FILL_NEG_INF,
                -1, 1.0f, softmax_x_desc, softmax_x,
                _opaque->mask_desc, _opaque->mask,
                softmax_y_desc, softmax_y));
            if (_opaque->unpack_y) {
                CHECK_BANG(cnnlCopy_v2(
                    cnnl_handle, _opaque->contiguous_y_desc, softmax_y,
                    _opaque->y_desc, y,
                    copy_workspace, _opaque->copy_workspace_size));
            }
            return INFINI_STATUS_SUCCESS;
        }));
    cnrtQueueSync(queue);
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::causal_softmax::bang
