#ifndef __CONV_NINETOOTHED_DESCRIPTOR_H__
#define __CONV_NINETOOTHED_DESCRIPTOR_H__

#include "../../../handle.h"
#include "../../../operator.h"
#include "../../../tensor.h"

#include "../../../../../build/ninetoothed/conv2d.h"
#include "../../../ninetoothed/utils.h"

#include "../../../operator.h"
#include "../../conv/info.h"
#include <assert.h>

// #include <cuda_runtime.h>

namespace op::conv::ninetoothed {

class Descriptor final : public InfiniopDescriptor {
public:
    using Size = ::ninetoothed::Tensor<>::Size;
    using Stride = ::ninetoothed::Tensor<>::Stride;

    std::vector<Stride> _y_strides;
    std::vector<Size> _y_shape;
    std::vector<Stride> _x_strides;
    std::vector<Size> _x_shape;
    std::vector<Stride> _w_strides;
    std::vector<Size> _w_shape;
    std::vector<Stride> _b_strides;
    std::vector<Size> _b_shape;

    int _kernel_size_h;
    int _kernel_size_w;
    ptrdiff_t _stride_h;
    ptrdiff_t _stride_w;
    size_t _padding_h;
    size_t _padding_w;
    size_t _dilation_h;
    size_t _dilation_w;
    int _ceil_mode;
    infiniDtype_t _dtype;

    Descriptor(
        infiniopHandle_t handle,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t w_desc,
        infiniopTensorDescriptor_t b_desc,
        ptrdiff_t stride_h,
        ptrdiff_t stride_w,
        size_t padding_h,
        size_t padding_w,
        size_t dilation_h,
        size_t dilation_w,
        int ceil_mode)
        : InfiniopDescriptor{handle->device, handle->device_id},
          _y_strides(y_desc->strides()), _y_shape(y_desc->shape()),
          _x_strides(x_desc->strides()), _x_shape(x_desc->shape()),
          _w_strides(w_desc->strides()), _w_shape(w_desc->shape()),
          _b_strides(b_desc->strides()), _b_shape(b_desc->shape()),
          _kernel_size_h(w_desc->shape()[2]), _kernel_size_w(w_desc->shape()[3]),
          _stride_h(stride_h), _stride_w(stride_w),
          _padding_h(padding_h), _padding_w(padding_w),
          _dilation_h(dilation_h), _dilation_w(dilation_w),
          _ceil_mode(ceil_mode), _dtype(x_desc->dtype()) {}

public:
    ~Descriptor() = default;

    size_t workspaceSize() const { return 0; }

    static infiniStatus_t create(
        infiniopHandle_t handle_,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t w_desc,
        infiniopTensorDescriptor_t b_desc,
        const void *pads,
        const void *strides,
        const void *dilations,
        size_t n) {

        assert(2 == n);

        ptrdiff_t stride_h = reinterpret_cast<const ptrdiff_t *>(strides)[0];
        ptrdiff_t stride_w = reinterpret_cast<const ptrdiff_t *>(strides)[1];
        size_t padding_h = reinterpret_cast<const size_t *>(pads)[0];
        size_t padding_w = reinterpret_cast<const size_t *>(pads)[1];
        size_t dilation_h = reinterpret_cast<const size_t *>(dilations)[0];
        size_t dilation_w = reinterpret_cast<const size_t *>(dilations)[1];

        *desc_ptr = new Descriptor{handle_, y_desc, x_desc, w_desc, b_desc, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, 0};

        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t calculate(
        void *workspace, size_t workspace_size,
        void *y,
        const void *x,
        const void *w,
        const void *bias,
        void *stream) const {

        auto x_tensor{::ninetoothed::Tensor{x, _x_shape, _x_strides}};
        auto w_tensor{::ninetoothed::Tensor{w, _w_shape, _w_strides}};
        auto b_tensor{::ninetoothed::Tensor{bias, _b_shape, _b_strides}};
        auto y_tensor{::ninetoothed::Tensor{y, _y_shape, _y_strides}};
        auto input_precision = ::ninetoothed::Tensor{1};

        auto error_code = launch_conv2d(stream,
                                        x_tensor, w_tensor, b_tensor,
                                        y_tensor,
                                        input_precision,
                                        1,
                                        _stride_h, _stride_w,
                                        _padding_h, _padding_w,
                                        _dilation_h, _dilation_w,
                                        _dtype,
                                        32, 32, 32);
        if (error_code) {
            // printf("calculate conv2d descriptor      333: %d, %s\n", error_code, cudaGetErrorString(static_cast<cudaError_t>(error_code)));
            return INFINI_STATUS_NOT_IMPLEMENTED;
        }

        return INFINI_STATUS_SUCCESS;
    }
};

} // namespace op::conv::ninetoothed

#endif // __CONV_NINETOOTHED_DESCRIPTOR_H__