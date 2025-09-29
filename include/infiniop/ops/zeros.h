#ifndef __INFINIOP_ZEROS_API_H__
#define __INFINIOP_ZEROS_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopZerosDescriptor_t;

__C __export infiniStatus_t infiniopCreateZerosDescriptor(infiniopHandle_t handle,
                                                          infiniopZerosDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t y,
                                                          infiniopTensorDescriptor_t x);

__C __export infiniStatus_t infiniopGetZerosWorkspaceSize(infiniopZerosDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopZeros(infiniopZerosDescriptor_t desc,
                                          void *workspace,
                                          size_t workspace_size,
                                          void *y,
                                          const void *x,
                                          void *stream);

__C __export infiniStatus_t infiniopDestroyZerosDescriptor(infiniopZerosDescriptor_t desc);

#endif
