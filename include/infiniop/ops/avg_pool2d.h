#ifndef __INFINIOP_AVG_POOL2D_API_H__
#define __INFINIOP_AVG_POOL2D_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopAvgPool2dDescriptor_t;

__C __export infiniStatus_t infiniopCreateAvgPool2dDescriptor(
    infiniopHandle_t handle,
    infiniopAvgPool2dDescriptor_t *desc_ptr,
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
    int ceil_mode);

__C __export infiniStatus_t infiniopGetAvgPool2dWorkspaceSize(
    infiniopAvgPool2dDescriptor_t desc,
    size_t *size);

__C __export infiniStatus_t infiniopAvgPool2d(
    infiniopAvgPool2dDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream);

__C __export infiniStatus_t infiniopDestroyAvgPool2dDescriptor(
    infiniopAvgPool2dDescriptor_t desc);

#endif
