#include "gelu_bang.h"
#include "../../../devices/bang/common_bang.h"

#include <algorithm>

namespace op::gelu::bang {

static size_t alignUp(size_t value, size_t alignment = 128) {
    return (value + alignment - 1) / alignment * alignment;
}

struct Descriptor::Opaque {
    std::shared_ptr<device::bang::Handle::Internal> internal;
    cnnlActivationDescriptor_t activation_desc = nullptr;
    cnnlTensorDescriptor_t x_desc = nullptr;
    cnnlTensorDescriptor_t y_desc = nullptr;
    cnnlTensorDescriptor_t contiguous_x_desc = nullptr;
    cnnlTensorDescriptor_t contiguous_y_desc = nullptr;
    size_t contiguous_x_offset = 0;
    size_t contiguous_y_offset = 0;
    size_t copy_workspace_offset = 0;
    size_t copy_workspace_size = 0;
    bool pack_x = false;
    bool unpack_y = false;

    ~Opaque() {
        if (activation_desc) {
            cnnlDestroyActivationDescriptor(activation_desc);
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
    }
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    if (input_desc_vec.size() != 1) {
        return INFINI_STATUS_BAD_PARAM;
    }
    auto handle = reinterpret_cast<device::bang::Handle *>(handle_);
    const auto dtype = out_desc->dtype();
    const auto &input_desc = input_desc_vec.at(0);
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
    CHECK_OR_RETURN(input_desc->dtype() == dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_SAME_SHAPE(out_desc->shape(), input_desc->shape());
    CHECK_OR_RETURN(!out_desc->hasBroadcastDim(), INFINI_STATUS_BAD_TENSOR_STRIDES);

    auto opaque = new Opaque{handle->internal()};
    CHECK_BANG(cnnlCreateActivationDescriptor(&opaque->activation_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->x_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->y_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->contiguous_x_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->contiguous_y_desc));
    CHECK_STATUS(device::bang::setCnnlTensorEx(opaque->x_desc, input_desc));
    CHECK_STATUS(device::bang::setCnnlTensorEx(opaque->y_desc, out_desc));
    CHECK_STATUS(device::bang::setCnnlTensor(opaque->contiguous_x_desc, input_desc));
    CHECK_STATUS(device::bang::setCnnlTensor(opaque->contiguous_y_desc, out_desc));
    opaque->pack_x = !input_desc->isContiguous();
    opaque->unpack_y = !out_desc->isContiguous();

    cnnlActivationMode_t mode = CNNL_ACTIVATION_GELU;
    cnnlComputationPreference_t preference = CNNL_COMPUTATION_HIGH_PRECISION;
    cnnlNanPropagation_t nan = CNNL_PROPAGATE_NAN;
    bool approximate = false;
    CHECK_BANG(cnnlSetActivationDescAttr(
        opaque->activation_desc, CNNL_ACTIVATION_MODE, &mode, sizeof(mode)));
    CHECK_BANG(cnnlSetActivationDescAttr(
        opaque->activation_desc, CNNL_ACTIVATION_PREFERENCE,
        &preference, sizeof(preference)));
    CHECK_BANG(cnnlSetActivationDescAttr(
        opaque->activation_desc, CNNL_ACTIVATION_NAN_PROP, &nan, sizeof(nan)));
    CHECK_BANG(cnnlSetActivationDescAttr(
        opaque->activation_desc, CNNL_ACTIVATION_APPROXIMATE,
        &approximate, sizeof(approximate)));

    CHECK_STATUS(opaque->internal->useCnnl(
        nullptr,
        [&](cnnlHandle_t cnnl_handle) {
            size_t copy_size = 0;
            if (opaque->pack_x) {
                CHECK_BANG(cnnlGetCopyWorkspaceSize(
                    cnnl_handle, opaque->x_desc, opaque->contiguous_x_desc,
                    &copy_size));
                opaque->copy_workspace_size = std::max(opaque->copy_workspace_size, copy_size);
            }
            if (opaque->unpack_y) {
                CHECK_BANG(cnnlGetCopyWorkspaceSize(
                    cnnl_handle, opaque->contiguous_y_desc, opaque->y_desc,
                    &copy_size));
                opaque->copy_workspace_size = std::max(opaque->copy_workspace_size, copy_size);
            }
            return INFINI_STATUS_SUCCESS;
        }));

    size_t workspace_size = 0;
    const size_t tensor_bytes = out_desc->numel() * infiniSizeOf(dtype);
    if (opaque->pack_x) {
        opaque->contiguous_x_offset = workspace_size;
        workspace_size += alignUp(tensor_bytes);
    }
    if (opaque->unpack_y) {
        opaque->contiguous_y_offset = workspace_size;
        workspace_size += alignUp(tensor_bytes);
    }
    opaque->copy_workspace_offset = workspace_size;
    workspace_size += alignUp(opaque->copy_workspace_size);

    *desc_ptr = new Descriptor(
        opaque, workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *output, std::vector<const void *> inputs,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    if (inputs.size() != 1) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto base = reinterpret_cast<char *>(workspace);
    const void *activation_x = inputs[0];
    void *activation_y = output;
    auto activation_x_desc = _opaque->x_desc;
    auto activation_y_desc = _opaque->y_desc;
    if (_opaque->pack_x) {
        activation_x = base + _opaque->contiguous_x_offset;
        activation_x_desc = _opaque->contiguous_x_desc;
    }
    if (_opaque->unpack_y) {
        activation_y = base + _opaque->contiguous_y_offset;
        activation_y_desc = _opaque->contiguous_y_desc;
    }
    void *copy_workspace = _opaque->copy_workspace_size == 0
                             ? nullptr
                             : base + _opaque->copy_workspace_offset;
    auto queue = reinterpret_cast<cnrtQueue_t>(stream);
    CHECK_STATUS(_opaque->internal->useCnnl(
        queue,
        [&](cnnlHandle_t cnnl_handle) {
            if (_opaque->pack_x) {
                CHECK_BANG(cnnlCopy_v2(
                    cnnl_handle, _opaque->x_desc, inputs[0],
                    _opaque->contiguous_x_desc, const_cast<void *>(activation_x),
                    copy_workspace, _opaque->copy_workspace_size));
            }
            CHECK_BANG(cnnlActivationForward(
                cnnl_handle, _opaque->activation_desc, nullptr,
                activation_x_desc, activation_x, nullptr,
                activation_y_desc, activation_y));
            if (_opaque->unpack_y) {
                CHECK_BANG(cnnlCopy_v2(
                    cnnl_handle, _opaque->contiguous_y_desc, activation_y,
                    _opaque->y_desc, output,
                    copy_workspace, _opaque->copy_workspace_size));
            }
            return INFINI_STATUS_SUCCESS;
        }));
    cnrtQueueSync(queue);
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gelu::bang
