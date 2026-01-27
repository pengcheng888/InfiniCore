#ifndef __AVG_POOL2D_NINETOOTHED_DESCRIPTOR_H__
#define __AVG_POOL2D_NINETOOTHED_DESCRIPTOR_H__

#include "../../../handle.h"
#include "../../../operator.h"
#include "../../../tensor.h"

#include "../../../../../build/ninetoothed/avg_pool2d.h"
#include "../../../ninetoothed/utils.h"

namespace op::avg_pool2d::ninetoothed {

class Descriptor final : public InfiniopDescriptor {
public:
    Descriptor(infiniopHandle_t handle,
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
               int ceil_mode) : InfiniopDescriptor{handle->device, handle->device_id},
                                _output_strides(output_desc->strides()), _output_shape(output_desc->shape()),
                                _input_strides(input_desc->strides()), _input_shape(input_desc->shape()),
                                _kernel_size_h(kernel_size_h), _kernel_size_w(kernel_size_w),
                                _stride_h(stride_h), _stride_w(stride_w),
                                _padding_h(padding_h), _padding_w(padding_w),
                                _dilation_h(dilation_h), _dilation_w(dilation_w),
                                _ceil_mode(ceil_mode), _dtype(input_desc->dtype()) {}

    ~Descriptor() = default;

    size_t get_workspace_size() const {
        return 0;
    }

    infiniStatus_t calculate(void *workspace,
                             size_t workspace_size,
                             void *output,
                             const void *input,
                             void *stream) const {
        auto input_tensor{::ninetoothed::Tensor{input, _input_shape, _input_strides}};
        auto output_tensor{::ninetoothed::Tensor{output, _output_shape, _output_strides}};
        constexpr auto block_size{128};
        if (launch_avg_pool2d(stream,
                              input_tensor, output_tensor,
                              _kernel_size_h, _kernel_size_w,
                              _stride_h, _stride_w,
                              _padding_h, _padding_w,
                              _dilation_h, _dilation_w,
                              _ceil_mode, _dtype, block_size)) {
            return INFINI_STATUS_NOT_IMPLEMENTED;
        }
        return INFINI_STATUS_SUCCESS;
    }

    static infiniStatus_t create(infiniopHandle_t handle,
                                 Descriptor **desc,
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
        *desc = new Descriptor{handle, output_desc, input_desc, kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, ceil_mode};
        return INFINI_STATUS_SUCCESS;
    }

private:
    using Size = ::ninetoothed::Tensor<>::Size;
    using Stride = ::ninetoothed::Tensor<>::Stride;
    std::vector<Stride> _output_strides;
    std::vector<Size> _output_shape;
    std::vector<Stride> _input_strides;
    std::vector<Size> _input_shape;
    int _kernel_size_h;
    int _kernel_size_w;
    int _stride_h;
    int _stride_w;
    int _padding_h;
    int _padding_w;
    int _dilation_h;
    int _dilation_w;
    int _ceil_mode;
    infiniDtype_t _dtype;
};

} // namespace op::avg_pool2d::ninetoothed

#endif // __AVG_POOL2D_NINETOOTHED_DESCRIPTOR_H__
