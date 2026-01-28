#include "avg_pool2d_cpu.h"
#include "../../../../utils.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace op::avg_pool2d::cpu {

Descriptor::Descriptor(infiniopHandle_t handle,
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

size_t Descriptor::get_workspace_size() const {
    return 0;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
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
    *desc_ptr = new Descriptor{handle, output_desc, input_desc, kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, ceil_mode};
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
infiniStatus_t avg_pool2d_cpu_impl(
    size_t N, size_t C, size_t H, size_t W,
    size_t out_h, size_t out_w,
    ptrdiff_t input_stride_n, ptrdiff_t input_stride_c,
    ptrdiff_t input_stride_h, ptrdiff_t input_stride_w,
    ptrdiff_t output_stride_n, ptrdiff_t output_stride_c,
    ptrdiff_t output_stride_h, ptrdiff_t output_stride_w,
    int kernel_size_h, int kernel_size_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    T *output,
    const T *input) {
    // For average pooling, padding value is 0
    T zero_val;
    if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
        zero_val = utils::cast<T>(0.0f);
    } else {
        zero_val = static_cast<T>(0);
    }

#pragma omp parallel for collapse(2)
    for (ptrdiff_t n = 0; n < static_cast<ptrdiff_t>(N); ++n) {
        for (ptrdiff_t c = 0; c < static_cast<ptrdiff_t>(C); ++c) {
            const T *input_base = input + n * input_stride_n + c * input_stride_c;
            T *output_base = output + n * output_stride_n + c * output_stride_c;

            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    // Calculate input window start position
                    const int h_start = static_cast<int>(oh) * stride_h - padding_h;
                    const int w_start = static_cast<int>(ow) * stride_w - padding_w;

                    // Accumulate sum and count for average calculation
                    float sum = 0.0f;
                    int count = 0;

                    // Iterate over kernel window
                    for (int kh = 0; kh < kernel_size_h; ++kh) {
                        for (int kw = 0; kw < kernel_size_w; ++kw) {
                            // Calculate actual input position with dilation
                            const int h_idx = h_start + kh * dilation_h;
                            const int w_idx = w_start + kw * dilation_w;

                            // Check if position is within bounds
                            if (h_idx >= 0 && h_idx < static_cast<int>(H) && w_idx >= 0 && w_idx < static_cast<int>(W)) {
                                const T val = input_base[h_idx * input_stride_h + w_idx * input_stride_w];
                                if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
                                    sum += utils::cast<float>(val);
                                } else {
                                    sum += static_cast<float>(val);
                                }
                                count++;
                            }
                            // If out of bounds, padding value (0) doesn't contribute to sum
                        }
                    }

                    // Calculate average
                    T avg_val;
                    if (count > 0) {
                        float avg = sum / count;
                        if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
                            avg_val = utils::cast<T>(avg);
                        } else {
                            avg_val = static_cast<T>(avg);
                        }
                    } else {
                        avg_val = zero_val;
                    }

                    output_base[oh * output_stride_h + ow * output_stride_w] = avg_val;
                }
            }
        }
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {
    const size_t N = _input_shape[0];
    const size_t C = _input_shape[1];
    const size_t H = _input_shape[2];
    const size_t W = _input_shape[3];
    const size_t out_h = _output_shape[2];
    const size_t out_w = _output_shape[3];

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return avg_pool2d_cpu_impl<fp16_t>(
            N, C, H, W, out_h, out_w,
            _input_strides[0], _input_strides[1], _input_strides[2], _input_strides[3],
            _output_strides[0], _output_strides[1], _output_strides[2], _output_strides[3],
            _kernel_size_h, _kernel_size_w,
            _stride_h, _stride_w,
            _padding_h, _padding_w,
            _dilation_h, _dilation_w,
            reinterpret_cast<fp16_t *>(output),
            reinterpret_cast<const fp16_t *>(input));
    case INFINI_DTYPE_F32:
        return avg_pool2d_cpu_impl<float>(
            N, C, H, W, out_h, out_w,
            _input_strides[0], _input_strides[1], _input_strides[2], _input_strides[3],
            _output_strides[0], _output_strides[1], _output_strides[2], _output_strides[3],
            _kernel_size_h, _kernel_size_w,
            _stride_h, _stride_w,
            _padding_h, _padding_w,
            _dilation_h, _dilation_w,
            reinterpret_cast<float *>(output),
            reinterpret_cast<const float *>(input));
    case INFINI_DTYPE_BF16:
        return avg_pool2d_cpu_impl<bf16_t>(
            N, C, H, W, out_h, out_w,
            _input_strides[0], _input_strides[1], _input_strides[2], _input_strides[3],
            _output_strides[0], _output_strides[1], _output_strides[2], _output_strides[3],
            _kernel_size_h, _kernel_size_w,
            _stride_h, _stride_w,
            _padding_h, _padding_w,
            _dilation_h, _dilation_w,
            reinterpret_cast<bf16_t *>(output),
            reinterpret_cast<const bf16_t *>(input));
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_INTERNAL_ERROR;
}
} // namespace op::avg_pool2d::cpu
