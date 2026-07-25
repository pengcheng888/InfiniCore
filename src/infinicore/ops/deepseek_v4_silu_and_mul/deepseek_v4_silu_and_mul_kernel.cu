#include "deepseek_v4_silu_and_mul_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace infinicore::op::deepseek_v4_silu_and_mul_impl {
namespace {

constexpr int kThreads = 256;

template <typename T>
__device__ __forceinline__ float to_float(T v);

template <>
__device__ __forceinline__ float to_float<__half>(__half v) {
    return __half2float(v);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T from_float(float v);

template <>
__device__ __forceinline__ __half from_float<__half>(float v) {
    return __float2half(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <typename T>
__global__ void silu_and_mul_kernel(T *__restrict__ out,
                                    const T *__restrict__ x,
                                    int64_t tokens,
                                    int64_t hidden) {
    const int64_t token = blockIdx.x;
    if (token >= tokens) {
        return;
    }
    const int64_t input_base = token * hidden * 2;
    const int64_t output_base = token * hidden;
    for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float gate = to_float<T>(x[input_base + i]);
        const float up = to_float<T>(x[input_base + hidden + i]);
        const float activated = gate / (1.0f + expf(-gate));
        out[output_base + i] = from_float<T>(activated * up);
    }
}

} // namespace

void launch_silu_and_mul(void *out,
                         const void *x,
                         int64_t tokens,
                         int64_t hidden,
                         DataType dtype,
                         void *stream) {
    if (tokens <= 0 || hidden <= 0) {
        return;
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const dim3 grid(static_cast<unsigned int>(tokens));
    const dim3 block(kThreads);
    if (dtype == DataType::BF16) {
        silu_and_mul_kernel<__nv_bfloat16><<<grid, block, 0, cuda_stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(x),
            tokens,
            hidden);
    } else if (dtype == DataType::F16) {
        silu_and_mul_kernel<__half><<<grid, block, 0, cuda_stream>>>(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const __half *>(x),
            tokens,
            hidden);
    }
}

} // namespace infinicore::op::deepseek_v4_silu_and_mul_impl
