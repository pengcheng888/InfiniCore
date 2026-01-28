#ifndef __AVG_POOL2D_H__
#define __AVG_POOL2D_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include <vector>

#define DESCRIPTOR(NAMESPACE)                              \
                                                           \
    namespace op::avg_pool2d::NAMESPACE {                  \
    class Descriptor final : public InfiniopDescriptor {   \
    protected:                                             \
        using Size = std::size_t;                          \
        using Stride = std::ptrdiff_t;                     \
        std::vector<Stride> _output_strides;               \
        std::vector<Size> _output_shape;                   \
        std::vector<Stride> _input_strides;                \
        std::vector<Size> _input_shape;                    \
        int _kernel_size_h;                                \
        int _kernel_size_w;                                \
        int _stride_h;                                     \
        int _stride_w;                                     \
        int _padding_h;                                    \
        int _padding_w;                                    \
        int _dilation_h;                                   \
        int _dilation_w;                                   \
        int _ceil_mode;                                    \
        infiniDtype_t _dtype;                              \
        Descriptor(infiniopHandle_t handle,                \
                   infiniopTensorDescriptor_t output_desc, \
                   infiniopTensorDescriptor_t input_desc,  \
                   int kernel_size_h,                      \
                   int kernel_size_w,                      \
                   int stride_h,                           \
                   int stride_w,                           \
                   int padding_h,                          \
                   int padding_w,                          \
                   int dilation_h,                         \
                   int dilation_w,                         \
                   int ceil_mode);                         \
                                                           \
    public:                                                \
        ~Descriptor() = default;                           \
                                                           \
        size_t get_workspace_size() const;                 \
                                                           \
        static infiniStatus_t create(                      \
            infiniopHandle_t handle,                       \
            Descriptor **desc_ptr,                         \
            infiniopTensorDescriptor_t output_desc,        \
            infiniopTensorDescriptor_t input_desc,         \
            int kernel_size_h,                             \
            int kernel_size_w,                             \
            int stride_h,                                  \
            int stride_w,                                  \
            int padding_h,                                 \
            int padding_w,                                 \
            int dilation_h,                                \
            int dilation_w,                                \
            int ceil_mode);                                \
                                                           \
        infiniStatus_t calculate(                          \
            void *workspace, size_t workspace_size,        \
            void *output,                                  \
            const void *input,                             \
            void *stream) const;                           \
    };                                                     \
    }

#endif // __AVG_POOL2D_H__
