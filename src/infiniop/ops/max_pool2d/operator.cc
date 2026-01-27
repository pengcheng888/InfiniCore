#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/max_pool2d.h"

#ifdef ENABLE_CPU_API
// #include "cpu/max_pool2d_cpu.h"
#endif
#if defined(ENABLE_NINETOOTHED)
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_MOORE_API) || defined(ENABLE_METAX_API) || defined(ENABLE_IlUVATAR_API) || defined(ENABLE_HYGON_API)
#include "ninetoothed/descriptor.h"
#endif
#endif

__C infiniStatus_t infiniopCreateMaxPool2dDescriptor(
    infiniopHandle_t handle,
    infiniopMaxPool2dDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    int kernel_size_h,
    int kernel_size_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int ceil_mode) {

#define CREATE(CASE, NAMESPACE)                                                   \
    case CASE:                                                                    \
        return op::max_pool2d::NAMESPACE::Descriptor::create(                     \
            handle,                                                               \
            reinterpret_cast<op::max_pool2d::NAMESPACE::Descriptor **>(desc_ptr), \
            output_desc,                                                          \
            input_desc,                                                           \
            kernel_size_h,                                                        \
            kernel_size_w,                                                        \
            stride_h,                                                             \
            stride_w,                                                             \
            padding_h,                                                            \
            padding_w,                                                            \
            dilation_h,                                                           \
            dilation_w,                                                           \
            ceil_mode);

    switch (handle->device) {

#if defined(ENABLE_NINETOOTHED)
#if defined(ENABLE_NVIDIA_API)
        CREATE(INFINI_DEVICE_NVIDIA, ninetoothed);
#endif

#if defined(ENABLE_MOORE_API)
        CREATE(INFINI_DEVICE_MOORE, ninetoothed);
#endif

#if defined(ENABLE_METAX_API)
        CREATE(INFINI_DEVICE_METAX, ninetoothed);
#endif

#if defined(ENABLE_IlUVATAR_API)
        CREATE(INFINI_DEVICE_ILUVATAR, ninetoothed);
#endif

#if defined(ENABLE_HYGON_API)
        CREATE(INFINI_DEVICE_HYGON, ninetoothed);
#endif
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__C infiniStatus_t infiniopGetMaxPool2dWorkspaceSize(
    infiniopMaxPool2dDescriptor_t desc,
    size_t *size) {

#define GET_SIZE(CASE, NAMESPACE)                                                     \
    case CASE:                                                                        \
        *size = reinterpret_cast<const op::max_pool2d::NAMESPACE::Descriptor *>(desc) \
                    ->get_workspace_size();                                           \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#if defined(ENABLE_NINETOOTHED)
#if defined(ENABLE_NVIDIA_API)
        GET_SIZE(INFINI_DEVICE_NVIDIA, ninetoothed);
#endif
#if defined(ENABLE_MOORE_API)
        GET_SIZE(INFINI_DEVICE_MOORE, ninetoothed);
#endif
#if defined(ENABLE_METAX_API)
        GET_SIZE(INFINI_DEVICE_METAX, ninetoothed);
#endif
#if defined(ENABLE_IlUVATAR_API)
        GET_SIZE(INFINI_DEVICE_ILUVATAR, ninetoothed);
#endif
#if defined(ENABLE_HYGON_API)
        GET_SIZE(INFINI_DEVICE_HYGON, ninetoothed);
#endif
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET_SIZE
}

__C infiniStatus_t infiniopMaxPool2d(
    infiniopMaxPool2dDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                   \
    case CASE:                                                                       \
        return reinterpret_cast<const op::max_pool2d::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, output, input, stream);

    switch (desc->device_type) {

#if defined(ENABLE_NINETOOTHED)
#if defined(ENABLE_NVIDIA_API)
        CALCULATE(INFINI_DEVICE_NVIDIA, ninetoothed);
#endif
#if defined(ENABLE_MOORE_API)
        CALCULATE(INFINI_DEVICE_MOORE, ninetoothed);
#endif
#if defined(ENABLE_METAX_API)
        CALCULATE(INFINI_DEVICE_METAX, ninetoothed);
#endif
#if defined(ENABLE_IlUVATAR_API)
        CALCULATE(INFINI_DEVICE_ILUVATAR, ninetoothed);
#endif
#if defined(ENABLE_HYGON_API)
        CALCULATE(INFINI_DEVICE_HYGON, ninetoothed);
#endif
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t infiniopDestroyMaxPool2dDescriptor(
    infiniopMaxPool2dDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                \
    case CASE:                                                                  \
        delete reinterpret_cast<op::max_pool2d::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#if defined(ENABLE_NINETOOTHED)
#if defined(ENABLE_NVIDIA_API)
        DESTROY(INFINI_DEVICE_NVIDIA, ninetoothed);
#endif
#if defined(ENABLE_MOORE_API)
        DESTROY(INFINI_DEVICE_MOORE, ninetoothed);
#endif
#if defined(ENABLE_METAX_API)
        DESTROY(INFINI_DEVICE_METAX, ninetoothed);
#endif
#if defined(ENABLE_IlUVATAR_API)
        DESTROY(INFINI_DEVICE_ILUVATAR, ninetoothed);
#endif
#if defined(ENABLE_HYGON_API)
        DESTROY(INFINI_DEVICE_HYGON, ninetoothed);
#endif
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}