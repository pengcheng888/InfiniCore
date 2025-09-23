#ifndef __INFINIOP_EYES_API_H__
#define __INFINIOP_EYES_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopEyesDescriptor_t;

__C __export infiniStatus_t infiniopCreateEyesDescriptor(infiniopHandle_t handle,
                                                         infiniopEyesDescriptor_t *desc_ptr,
                                                         infiniopTensorDescriptor_t y,
                                                         infiniopTensorDescriptor_t x);

__C __export infiniStatus_t infiniopGetEyesWorkspaceSize(infiniopEyesDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopEyes(infiniopEyesDescriptor_t desc,
                                         void *workspace,
                                         size_t workspace_size,
                                         void *y,
                                         const void *x,
                                         void *stream);

__C __export infiniStatus_t infiniopDestroyEyesDescriptor(infiniopEyesDescriptor_t desc);

#endif
