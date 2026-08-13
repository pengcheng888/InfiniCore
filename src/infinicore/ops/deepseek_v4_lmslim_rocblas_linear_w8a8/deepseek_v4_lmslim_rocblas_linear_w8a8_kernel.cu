#include "deepseek_v4_lmslim_rocblas_linear_w8a8_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace infinicore::op::deepseek_v4_lmslim_rocblas_linear_w8a8_impl {
namespace {

constexpr int kBlockSize = 256;

template <typename T>
__device__ __forceinline__ T from_float(float value);

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <>
__device__ __forceinline__ __half from_float<__half>(float value) {
    return __float2half(value);
}

template <typename T>
__device__ __forceinline__ float bias_to_float(const T *bias, int64_t col);

template <>
__device__ __forceinline__ float bias_to_float<__nv_bfloat16>(const __nv_bfloat16 *bias, int64_t col) {
    return __bfloat162float(bias[col]);
}

template <>
__device__ __forceinline__ float bias_to_float<__half>(const __half *bias, int64_t col) {
    return __half2float(bias[col]);
}

template <typename T>
__global__ void apply_scales_kernel(T *__restrict__ output,
                                    const int32_t *__restrict__ accum,
                                    const float *__restrict__ input_scale,
                                    const float *__restrict__ weight_scale,
                                    const T *__restrict__ bias,
                                    int64_t total,
                                    int64_t n) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int64_t row = idx / n;
    const int64_t col = idx - row * n;
    float value = static_cast<float>(accum[idx]) * input_scale[row] * weight_scale[col];
    if (bias != nullptr) {
        value += bias_to_float<T>(bias, col);
    }
    output[idx] = from_float<T>(value);
}

} // namespace

void launch_apply_scales(void *output,
                         const int32_t *accum,
                         const float *input_scale,
                         const float *weight_scale,
                         const void *bias,
                         int64_t m,
                         int64_t n,
                         infinicore::DataType output_dtype,
                         void *stream) {
    const int64_t total = m * n;
    if (total <= 0) {
        return;
    }
    const dim3 block(kBlockSize);
    const dim3 grid(static_cast<unsigned int>((total + kBlockSize - 1) / kBlockSize));
    if (output_dtype == infinicore::DataType::BF16) {
        apply_scales_kernel<__nv_bfloat16><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<__nv_bfloat16 *>(output),
            accum,
            input_scale,
            weight_scale,
            reinterpret_cast<const __nv_bfloat16 *>(bias),
            total,
            n);
    } else {
        apply_scales_kernel<__half><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<__half *>(output),
            accum,
            input_scale,
            weight_scale,
            reinterpret_cast<const __half *>(bias),
            total,
            n);
    }
}

} // namespace infinicore::op::deepseek_v4_lmslim_rocblas_linear_w8a8_impl
