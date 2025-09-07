#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/topksoftmax.h"

#ifdef ENABLE_CPU_API
#include "cpu/topksoftmax_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API)
#include "nvidia/topksoftmax_nvidia.cuh"
#endif

__C infiniStatus_t infiniopCreateTopksoftmaxDescriptor(infiniopHandle_t handle,
                                                       infiniopTopksoftmaxDescriptor_t *desc_ptr,
                                                       infiniopTensorDescriptor_t x_desc) {

#define CREATE(CASE, NAMESPACE)                                \
    case CASE:                                                 \
        return op::topksoftmax::NAMESPACE::Descriptor::create( \
            handle, reinterpret_cast<op::topksoftmax::NAMESPACE::Descriptor **>(desc_ptr), x_desc)

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    }

#undef CREATE

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopGetTopksoftmaxWorkspaceSize(infiniopTopksoftmaxDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                       \
    case CASE:                                                                                     \
        *size = reinterpret_cast<op::topksoftmax::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    }

#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopTopksoftmax(infiniopTopksoftmaxDescriptor_t desc, void *workspace, size_t workspace_size,
                                       void *values, void *indices, const void *x, const size_t topk, const bool norm,
                                       void *stream) {
    if (topk > 32) {
        return INFINI_STATUS_BAD_PARAM;
    }

#define CALCULATE(CASE, NAMESPACE)                                                          \
    case CASE:                                                                              \
        return reinterpret_cast<op::topksoftmax::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, (float *)values, (int *)indices, x, topk, norm, stream)

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    }

#undef CALCULATE

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopDestroyTopksoftmaxDescriptor(infiniopTopksoftmaxDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                 \
    case CASE:                                                                   \
        delete reinterpret_cast<op::topksoftmax::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DESTROY(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    }

#undef DESTROY

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
