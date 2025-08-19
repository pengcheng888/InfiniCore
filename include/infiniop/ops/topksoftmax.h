#ifndef __INFINIOP_TOPKSOFTMAX_API_H__
#define __INFINIOP_TOPKSOFTMAX_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopTopksoftmaxDescriptor_t;

__C __export infiniStatus_t infiniopCreateTopksoftmaxDescriptor(
    infiniopHandle_t handle,
    infiniopTopksoftmaxDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc);

__C __export infiniStatus_t infiniopGetTopksoftmaxWorkspaceSize(infiniopTopksoftmaxDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopTopksoftmax(infiniopTopksoftmaxDescriptor_t desc, void *workspace, size_t workspace_size,
                                                void *values, void *indices, void *x, void *stream);

__C __export infiniStatus_t infiniopDestroyTopksoftmaxDescriptor(infiniopTopksoftmaxDescriptor_t desc);

#endif
