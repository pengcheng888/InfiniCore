#include "deepseek_v4_fused_experts_impl_int8_marlin_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace infinicore::op::deepseek_v4_fused_experts_impl_int8_marlin {
namespace {

constexpr int kBlockSize = 256;

__global__ void per_token_quant_int8_bf16_kernel(int8_t *__restrict__ output,
                                                 float *__restrict__ scale,
                                                 const __nv_bfloat16 *__restrict__ input,
                                                 int64_t rows,
                                                 int64_t cols) {
    const int64_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    float absmax = 0.0f;
    const int64_t base = row * cols;
    for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value = __bfloat162float(input[base + col]);
        absmax = fmaxf(absmax, fabsf(value));
    }

    __shared__ float reduce[kBlockSize];
    reduce[threadIdx.x] = absmax;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float row_absmax = fmaxf(reduce[0], 1.0e-10f);
    const float row_scale = row_absmax / 127.0f;
    const float inv_scale = 127.0f / row_absmax;
    if (threadIdx.x == 0) {
        scale[row] = row_scale;
    }

    for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value = __bfloat162float(input[base + col]);
        float q = nearbyintf(value * inv_scale);
        q = fminf(127.0f, fmaxf(-128.0f, q));
        output[base + col] = static_cast<int8_t>(q);
    }
}

__global__ void moe_sum_scale_add_bf16_kernel(__nv_bfloat16 *__restrict__ output,
                                              const __nv_bfloat16 *__restrict__ input,
                                              const __nv_bfloat16 *__restrict__ shared_output,
                                              int64_t total,
                                              int64_t topk,
                                              int64_t hidden,
                                              float factor) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int64_t token = idx / hidden;
    const int64_t col = idx - token * hidden;
    const int64_t base = (token * topk * hidden) + col;
    float acc = 0.0f;
    if (topk == 6) {
#pragma unroll
        for (int k = 0; k < 6; ++k) {
            acc += __bfloat162float(input[base + static_cast<int64_t>(k) * hidden]);
        }
    } else {
        for (int64_t k = 0; k < topk; ++k) {
            acc += __bfloat162float(input[base + k * hidden]);
        }
    }
    acc = acc * factor + __bfloat162float(shared_output[idx]);
    output[idx] = __float2bfloat16(acc);
}

} // namespace

void launch_per_token_quant_int8_bf16(void *output,
                                      float *scale,
                                      const void *input,
                                      int64_t rows,
                                      int64_t cols,
                                      void *stream) {
    per_token_quant_int8_bf16_kernel<<<static_cast<unsigned int>(rows), kBlockSize, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<int8_t *>(output),
        scale,
        reinterpret_cast<const __nv_bfloat16 *>(input),
        rows,
        cols);
    return;
}

void launch_moe_sum_scale_add_bf16(void *output,
                                   const void *input,
                                   const void *shared_output,
                                   int64_t tokens,
                                   int64_t topk,
                                   int64_t hidden,
                                   float factor,
                                   void *stream) {
    const int64_t total = tokens * hidden;
    const int blocks = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
    moe_sum_scale_add_bf16_kernel<<<blocks, kBlockSize, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<__nv_bfloat16 *>(output),
        reinterpret_cast<const __nv_bfloat16 *>(input),
        reinterpret_cast<const __nv_bfloat16 *>(shared_output),
        total,
        topk,
        hidden,
        factor);
    return;
}

} // namespace infinicore::op::deepseek_v4_fused_experts_impl_int8_marlin
