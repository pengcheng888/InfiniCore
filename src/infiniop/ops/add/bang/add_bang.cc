#include "add_bang.h"
#include "../../../devices/bang/common_bang.h"

namespace op::add::bang {

struct Descriptor::Opaque {
    std::shared_ptr<device::bang::Handle::Internal> internal;
    cnnlOpTensorDescriptor_t op_desc = nullptr;
    cnnlTensorDescriptor_t a_desc = nullptr;
    cnnlTensorDescriptor_t b_desc = nullptr;
    cnnlTensorDescriptor_t c_desc = nullptr;

    ~Opaque() {
        if (op_desc) {
            cnnlDestroyOpTensorDescriptor(op_desc);
        }
        if (a_desc) {
            cnnlDestroyTensorDescriptor(a_desc);
        }
        if (b_desc) {
            cnnlDestroyTensorDescriptor(b_desc);
        }
        if (c_desc) {
            cnnlDestroyTensorDescriptor(c_desc);
        }
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    if (input_desc_vec.size() != 2) {
        return INFINI_STATUS_BAD_PARAM;
    }
    auto handle = reinterpret_cast<device::bang::Handle *>(handle_);
    const auto &a_desc = input_desc_vec.at(0);
    const auto &b_desc = input_desc_vec.at(1);
    auto info_result = AddInfo::create(out_desc, a_desc, b_desc);
    CHECK_RESULT(info_result);
    auto info = info_result.take();

    auto opaque = new Opaque{handle->internal()};
    CHECK_BANG(cnnlCreateOpTensorDescriptor(&opaque->op_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->a_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->b_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->c_desc));
    CHECK_STATUS(device::bang::setCnnlTensorEx(opaque->a_desc, a_desc));
    CHECK_STATUS(device::bang::setCnnlTensorEx(opaque->b_desc, b_desc));
    CHECK_STATUS(device::bang::setCnnlTensorEx(opaque->c_desc, out_desc));

    auto compute_type = device::bang::getCnnlDtype(info.dtype);
    CHECK_BANG(cnnlSetOpTensorDescriptor(
        opaque->op_desc, CNNL_OP_TENSOR_ADD, compute_type, CNNL_NOT_PROPAGATE_NAN));

    size_t workspace_size = 0;
    CHECK_STATUS(opaque->internal->useCnnl(
        nullptr,
        [&](cnnlHandle_t cnnl_handle) {
            CHECK_BANG(cnnlGetOpTensorWorkspaceSize(
                cnnl_handle, opaque->a_desc, opaque->b_desc, opaque->c_desc,
                &workspace_size));
            return INFINI_STATUS_SUCCESS;
        }));

    *desc_ptr = new Descriptor(
        opaque, std::move(info), workspace_size, handle->device, handle->device_id);

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

    float one_f = 1.0f;
    float zero_f = 0.0f;
    int32_t one_i32 = 1;
    int32_t zero_i32 = 0;
    int64_t one_i64 = 1;
    int64_t zero_i64 = 0;
    const void *one = _info.dtype == INFINI_DTYPE_I64
                        ? static_cast<const void *>(&one_i64)
                    : _info.dtype == INFINI_DTYPE_I32
                        ? static_cast<const void *>(&one_i32)
                        : static_cast<const void *>(&one_f);
    const void *zero = _info.dtype == INFINI_DTYPE_I64
                         ? static_cast<const void *>(&zero_i64)
                     : _info.dtype == INFINI_DTYPE_I32
                         ? static_cast<const void *>(&zero_i32)
                         : static_cast<const void *>(&zero_f);
    auto queue = reinterpret_cast<cnrtQueue_t>(stream);
    CHECK_STATUS(_opaque->internal->useCnnl(
        queue,
        [&](cnnlHandle_t cnnl_handle) {
            CHECK_BANG(cnnlOpTensor(
                cnnl_handle, _opaque->op_desc,
                one, _opaque->a_desc, inputs[0],
                one, _opaque->b_desc, inputs[1],
                workspace, _workspace_size,
                zero, _opaque->c_desc, output));
            return INFINI_STATUS_SUCCESS;
        }));
    cnrtQueueSync(queue);
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::add::bang
