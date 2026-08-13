#include "deepseek_v4_lightop_linear_w8a8_smooth_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace infinicore::op::deepseek_v4_lightop_linear_w8a8_smooth_impl {
namespace {

constexpr int kBlockSize = 256;

__global__ void w8a8_smooth_gemm_bf16_kernel(__nv_bfloat16 *__restrict__ output,
                                             const int8_t *__restrict__ q_input,
                                             const int8_t *__restrict__ weight,
                                             const float *__restrict__ input_scale,
                                             const float *__restrict__ weight_scale,
                                             const __nv_bfloat16 *__restrict__ bias,
                                             int64_t m,
                                             int64_t n,
                                             int64_t k) {
    const int64_t out_idx = static_cast<int64_t>(blockIdx.x);
    if (out_idx >= m * n) {
        return;
    }
    const int64_t row = out_idx / n;
    const int64_t col = out_idx - row * n;

    int acc = 0;
    const int64_t input_base = row * k;
    const int64_t weight_base = col * k;
    for (int64_t i = threadIdx.x; i < k; i += blockDim.x) {
        acc += static_cast<int>(q_input[input_base + i]) * static_cast<int>(weight[weight_base + i]);
    }

    __shared__ int reduce[kBlockSize];
    reduce[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float value = static_cast<float>(reduce[0]) * input_scale[row] * weight_scale[col];
        if (bias != nullptr) {
            value += __bfloat162float(bias[col]);
        }
        output[out_idx] = __float2bfloat16(value);
    }
}

} // namespace

void launch_w8a8_smooth_gemm_bf16(void *output,
                                  const int8_t *q_input,
                                  const int8_t *weight,
                                  const float *input_scale,
                                  const float *weight_scale,
                                  const void *bias,
                                  int64_t m,
                                  int64_t n,
                                  int64_t k,
                                  void *stream) {
    const int64_t total = m * n;
    if (total <= 0) {
        return;
    }
    w8a8_smooth_gemm_bf16_kernel<<<static_cast<unsigned int>(total),
                                    kBlockSize,
                                    0,
                                    reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<__nv_bfloat16 *>(output),
        q_input,
        weight,
        input_scale,
        weight_scale,
        reinterpret_cast<const __nv_bfloat16 *>(bias),
        m,
        n,
        k);
}

} // namespace infinicore::op::deepseek_v4_lightop_linear_w8a8_smooth_impl
