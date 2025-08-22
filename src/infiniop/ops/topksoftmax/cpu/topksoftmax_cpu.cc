#include "topksoftmax_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"

namespace op::topksoftmax::cpu {

Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    size_t N, size_t width, size_t topk) {

    return INFINI_STATUS_NOT_IMPLEMENTED;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    float *values, int *indices, void *x,
    void *stream) const {

    return INFINI_STATUS_NOT_IMPLEMENTED;
}
} // namespace op::topksoftmax::cpu
