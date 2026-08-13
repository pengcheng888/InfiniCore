#include "swiglu_bang.h"

#include "../../../devices/bang/common_bang.h"

#include <algorithm>

namespace op::swiglu::bang {

static size_t alignUp(size_t value, size_t alignment = 128) {
    return (value + alignment - 1) / alignment * alignment;
}

struct Descriptor::Opaque {
    std::shared_ptr<device::bang::Handle::Internal> internal;
    cnnlActivationDescriptor_t activation_desc = nullptr;
    cnnlOpTensorDescriptor_t mul_desc = nullptr;
    cnnlTensorDescriptor_t up_desc = nullptr;
    cnnlTensorDescriptor_t gate_desc = nullptr;
    cnnlTensorDescriptor_t output_desc = nullptr;
    cnnlTensorDescriptor_t contiguous_desc = nullptr;
    size_t temporary_offset = 0;
    size_t scratch_offset = 0;
    size_t copy_workspace_size = 0;
    size_t mul_workspace_size = 0;

    ~Opaque() {
        if (activation_desc) {
            cnnlDestroyActivationDescriptor(activation_desc);
        }
        if (mul_desc) {
            cnnlDestroyOpTensorDescriptor(mul_desc);
        }
        if (up_desc) {
            cnnlDestroyTensorDescriptor(up_desc);
        }
        if (gate_desc) {
            cnnlDestroyTensorDescriptor(gate_desc);
        }
        if (output_desc) {
            cnnlDestroyTensorDescriptor(output_desc);
        }
        if (contiguous_desc) {
            cnnlDestroyTensorDescriptor(contiguous_desc);
        }
    }
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    std::vector<infiniopTensorDescriptor_t> input_descs) {

    if (input_descs.size() != 2) {
        return INFINI_STATUS_BAD_PARAM;
    }
    const auto &up_desc = input_descs.at(0);
    const auto &gate_desc = input_descs.at(1);
    const auto dtype = output_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
    CHECK_OR_RETURN(up_desc->dtype() == dtype && gate_desc->dtype() == dtype,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_SAME_SHAPE(output_desc->shape(), up_desc->shape(), gate_desc->shape());
    CHECK_OR_RETURN(!output_desc->hasBroadcastDim(), INFINI_STATUS_BAD_TENSOR_STRIDES);

    auto handle = reinterpret_cast<device::bang::Handle *>(handle_);
    auto opaque = new Opaque{handle->internal()};
    CHECK_BANG(cnnlCreateActivationDescriptor(&opaque->activation_desc));
    CHECK_BANG(cnnlCreateOpTensorDescriptor(&opaque->mul_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->up_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->gate_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->output_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->contiguous_desc));
    CHECK_STATUS(device::bang::setCnnlTensorEx(opaque->up_desc, up_desc));
    CHECK_STATUS(device::bang::setCnnlTensorEx(opaque->gate_desc, gate_desc));
    CHECK_STATUS(device::bang::setCnnlTensorEx(opaque->output_desc, output_desc));
    CHECK_STATUS(device::bang::setCnnlTensor(opaque->contiguous_desc, output_desc));

    cnnlActivationMode_t mode = CNNL_ACTIVATION_SILU;
    cnnlComputationPreference_t preference = CNNL_COMPUTATION_HIGH_PRECISION;
    cnnlNanPropagation_t nan = CNNL_PROPAGATE_NAN;
    CHECK_BANG(cnnlSetActivationDescAttr(
        opaque->activation_desc, CNNL_ACTIVATION_MODE, &mode, sizeof(mode)));
    CHECK_BANG(cnnlSetActivationDescAttr(
        opaque->activation_desc, CNNL_ACTIVATION_PREFERENCE,
        &preference, sizeof(preference)));
    CHECK_BANG(cnnlSetActivationDescAttr(
        opaque->activation_desc, CNNL_ACTIVATION_NAN_PROP, &nan, sizeof(nan)));
    CHECK_BANG(cnnlSetOpTensorDescriptor(
        opaque->mul_desc, CNNL_OP_TENSOR_MUL,
        device::bang::getCnnlDtype(dtype), CNNL_NOT_PROPAGATE_NAN));

    CHECK_STATUS(opaque->internal->useCnnl(
        nullptr,
        [&](cnnlHandle_t cnnl_handle) {
            CHECK_BANG(cnnlGetCopyWorkspaceSize(
                cnnl_handle, opaque->gate_desc, opaque->contiguous_desc,
                &opaque->copy_workspace_size));
            CHECK_BANG(cnnlGetOpTensorWorkspaceSize(
                cnnl_handle, opaque->up_desc, opaque->contiguous_desc,
                opaque->output_desc, &opaque->mul_workspace_size));
            return INFINI_STATUS_SUCCESS;
        }));

    const size_t temporary_size = output_desc->numel() * infiniSizeOf(dtype);
    opaque->temporary_offset = 0;
    opaque->scratch_offset = alignUp(temporary_size);
    const size_t workspace_size = opaque->scratch_offset
                                + std::max(opaque->copy_workspace_size,
                                           opaque->mul_workspace_size);
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
    if (inputs.size() != 2) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto base = reinterpret_cast<char *>(workspace);
    void *temporary = base + _opaque->temporary_offset;
    void *scratch = base + _opaque->scratch_offset;
    auto queue = reinterpret_cast<cnrtQueue_t>(stream);
    float one = 1.0f;
    float zero = 0.0f;
    CHECK_STATUS(_opaque->internal->useCnnl(
        queue,
        [&](cnnlHandle_t cnnl_handle) {
            CHECK_BANG(cnnlCopy_v2(
                cnnl_handle, _opaque->gate_desc, inputs[1],
                _opaque->contiguous_desc, temporary,
                scratch, _opaque->copy_workspace_size));
            CHECK_BANG(cnnlActivationForward(
                cnnl_handle, _opaque->activation_desc, nullptr,
                _opaque->contiguous_desc, temporary, nullptr,
                _opaque->contiguous_desc, temporary));
            CHECK_BANG(cnnlOpTensor(
                cnnl_handle, _opaque->mul_desc,
                &one, _opaque->up_desc, inputs[0],
                &one, _opaque->contiguous_desc, temporary,
                scratch, _opaque->mul_workspace_size,
                &zero, _opaque->output_desc, output));
            return INFINI_STATUS_SUCCESS;
        }));
    cnrtQueueSync(queue);
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::swiglu::bang
