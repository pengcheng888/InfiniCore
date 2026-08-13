#include "rms_norm_bang.h"

#include "../../../devices/bang/common_bang.h"

namespace op::rms_norm::bang {

static size_t alignUp(size_t value, size_t alignment = 128) {
    return (value + alignment - 1) / alignment * alignment;
}

struct Descriptor::Opaque {
    std::shared_ptr<device::bang::Handle::Internal> internal;
    cnnlNormDescriptor_t norm_desc = nullptr;
    cnnlTensorDescriptor_t x_desc = nullptr;
    cnnlTensorDescriptor_t y_desc = nullptr;
    cnnlTensorDescriptor_t contiguous_x_desc = nullptr;
    cnnlTensorDescriptor_t contiguous_y_desc = nullptr;
    cnnlTensorDescriptor_t weight_desc = nullptr;
    cnnlTensorDescriptor_t cast_weight_desc = nullptr;
    cnnlTensorDescriptor_t rms_desc = nullptr;
    size_t cnnl_workspace_size = 0;
    size_t saved_rms_offset = 0;
    size_t cast_weight_offset = 0;
    size_t contiguous_x_offset = 0;
    size_t contiguous_y_offset = 0;
    size_t copy_workspace_offset = 0;
    size_t copy_workspace_size = 0;
    bool cast_weight = false;
    bool pack_x = false;
    bool unpack_y = false;
    cnnlCastDataType_t cast_type = CNNL_CAST_HALF_TO_FLOAT;

    ~Opaque() {
        if (norm_desc) {
            cnnlDestroyNormDesc(norm_desc);
        }
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
        if (weight_desc) {
            cnnlDestroyTensorDescriptor(weight_desc);
        }
        if (cast_weight_desc) {
            cnnlDestroyTensorDescriptor(cast_weight_desc);
        }
        if (rms_desc) {
            cnnlDestroyTensorDescriptor(rms_desc);
        }
    }
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    float epsilon) {

    auto result = RMSNormInfo::create(y_desc, x_desc, w_desc, epsilon);
    CHECK_RESULT(result);
    auto info = result.take();
    CHECK_OR_RETURN(info.atype == INFINI_DTYPE_F16
                        || info.atype == INFINI_DTYPE_BF16
                        || info.atype == INFINI_DTYPE_F32,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);

    auto handle = reinterpret_cast<device::bang::Handle *>(handle_);
    auto opaque = new Opaque{handle->internal()};
    CHECK_BANG(cnnlCreateNormDesc(&opaque->norm_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->x_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->y_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->contiguous_x_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->contiguous_y_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->weight_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->rms_desc));
    CHECK_STATUS(device::bang::setCnnlTensorEx(opaque->x_desc, x_desc));
    CHECK_STATUS(device::bang::setCnnlTensorEx(opaque->y_desc, y_desc));
    CHECK_STATUS(device::bang::setCnnlTensor(opaque->contiguous_x_desc, x_desc));
    CHECK_STATUS(device::bang::setCnnlTensor(opaque->contiguous_y_desc, y_desc));
    opaque->pack_x = !x_desc->isContiguous();
    opaque->unpack_y = !y_desc->isContiguous();
    CHECK_STATUS(device::bang::setCnnlTensor(opaque->weight_desc, w_desc));

    std::vector<int64_t> rms_dims(
        info.shape.begin(), info.shape.end() - 1);
    if (rms_dims.empty()) {
        rms_dims.push_back(1);
    }
    CHECK_BANG(cnnlSetTensorDescriptor_v2(
        opaque->rms_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT,
        static_cast<int>(rms_dims.size()), rms_dims.data()));

    const bool half_bfloat_cross = (info.atype == INFINI_DTYPE_F16 && info.wtype == INFINI_DTYPE_BF16)
                                || (info.atype == INFINI_DTYPE_BF16 && info.wtype == INFINI_DTYPE_F16);
    opaque->cast_weight = half_bfloat_cross;
    if (opaque->cast_weight) {
        opaque->cast_type = info.wtype == INFINI_DTYPE_F16
                              ? CNNL_CAST_HALF_TO_FLOAT
                              : CNNL_CAST_BFLOAT16_TO_FLOAT;
        CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->cast_weight_desc));
        int64_t weight_dim = static_cast<int64_t>(info.dim());
        CHECK_BANG(cnnlSetTensorDescriptor_v2(
            opaque->cast_weight_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT,
            1, &weight_dim));
    }

    CHECK_STATUS(opaque->internal->useCnnl(
        nullptr,
        [&](cnnlHandle_t cnnl_handle) {
            auto norm_x_desc = opaque->pack_x
                                 ? opaque->contiguous_x_desc
                                 : opaque->x_desc;
            CHECK_BANG(cnnlGetRmsNormOpWorkspaceSize(
                cnnl_handle, static_cast<int>(info.ndim() - 1),
                norm_x_desc, &opaque->cnnl_workspace_size));
            size_t copy_size = 0;
            if (opaque->pack_x) {
                CHECK_BANG(cnnlGetCopyWorkspaceSize(
                    cnnl_handle, opaque->x_desc, opaque->contiguous_x_desc, &copy_size));
                opaque->copy_workspace_size = std::max(opaque->copy_workspace_size, copy_size);
            }
            if (opaque->unpack_y) {
                CHECK_BANG(cnnlGetCopyWorkspaceSize(
                    cnnl_handle, opaque->contiguous_y_desc, opaque->y_desc, &copy_size));
                opaque->copy_workspace_size = std::max(opaque->copy_workspace_size, copy_size);
            }
            return INFINI_STATUS_SUCCESS;
        }));

    opaque->saved_rms_offset = alignUp(opaque->cnnl_workspace_size);
    size_t rows = 1;
    for (size_t i = 0; i + 1 < info.ndim(); ++i) {
        rows *= info.shape[i];
    }
    size_t workspace_size = opaque->saved_rms_offset + alignUp(rows * sizeof(float));
    if (opaque->cast_weight) {
        opaque->cast_weight_offset = workspace_size;
        workspace_size += alignUp(info.dim() * sizeof(float));
    }

    const size_t tensor_bytes = y_desc->numel() * infiniSizeOf(info.atype);
    if (opaque->pack_x) {
        opaque->contiguous_x_offset = workspace_size;
        workspace_size += alignUp(tensor_bytes);
    }
    if (opaque->unpack_y) {
        opaque->contiguous_y_offset = workspace_size;
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
    void *y, const void *x, const void *w,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto queue = reinterpret_cast<cnrtQueue_t>(stream);
    auto base = reinterpret_cast<char *>(workspace);
    void *saved_rms = base + _opaque->saved_rms_offset;
    const void *scale = w;
    auto scale_desc = _opaque->weight_desc;
    const void *norm_x = x;
    void *norm_y = y;
    auto norm_x_desc = _opaque->x_desc;
    auto norm_y_desc = _opaque->y_desc;
    if (_opaque->pack_x) {
        norm_x = base + _opaque->contiguous_x_offset;
        norm_x_desc = _opaque->contiguous_x_desc;
    }
    if (_opaque->unpack_y) {
        norm_y = base + _opaque->contiguous_y_offset;
        norm_y_desc = _opaque->contiguous_y_desc;
    }
    void *copy_workspace = _opaque->copy_workspace_size == 0
                             ? nullptr
                             : base + _opaque->copy_workspace_offset;

    CHECK_STATUS(_opaque->internal->useCnnl(
        queue,
        [&](cnnlHandle_t cnnl_handle) {
            if (_opaque->pack_x) {
                CHECK_BANG(cnnlCopy_v2(
                    cnnl_handle, _opaque->x_desc, x,
                    _opaque->contiguous_x_desc, const_cast<void *>(norm_x),
                    copy_workspace, _opaque->copy_workspace_size));
            }
            if (_opaque->cast_weight) {
                scale = base + _opaque->cast_weight_offset;
                scale_desc = _opaque->cast_weight_desc;
                CHECK_BANG(cnnlCastDataType(
                    cnnl_handle, _opaque->weight_desc, w, _opaque->cast_type,
                    _opaque->cast_weight_desc, const_cast<void *>(scale)));
            }
            CHECK_BANG(cnnlRmsNormForward_v2(
                cnnl_handle, static_cast<int>(_info.ndim() - 1), _info.epsilon,
                _opaque->norm_desc, norm_x_desc, norm_x,
                scale_desc, scale, nullptr,
                workspace, _opaque->cnnl_workspace_size,
                norm_y_desc, norm_y, _opaque->rms_desc, saved_rms));
            if (_opaque->unpack_y) {
                CHECK_BANG(cnnlCopy_v2(
                    cnnl_handle, _opaque->contiguous_y_desc, norm_y,
                    _opaque->y_desc, y,
                    copy_workspace, _opaque->copy_workspace_size));
            }
            return INFINI_STATUS_SUCCESS;
        }));
    cnrtQueueSync(queue);
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::rms_norm::bang
