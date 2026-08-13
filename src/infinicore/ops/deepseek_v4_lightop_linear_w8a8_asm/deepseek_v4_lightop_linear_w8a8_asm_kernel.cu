#include "deepseek_v4_lightop_linear_w8a8_asm_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace infinicore::op::deepseek_v4_lightop_linear_w8a8_asm_impl {
namespace {

constexpr int kThreads = 256;
constexpr int64_t kBlockSize = 128;

__global__ void expand_input_scale_kernel(float *__restrict__ input_block_scale,
                                          const float *__restrict__ input_scale,
                                          int64_t m,
                                          int64_t k_blocks) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = k_blocks * m;
    if (idx >= total) {
        return;
    }
    const int64_t token = idx % m;
    input_block_scale[idx] = input_scale[token];
}

__global__ void fill_weight_block_scale_kernel(float *__restrict__ weight_block_scale,
                                               int64_t total) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total) {
        weight_block_scale[idx] = 1.0f;
    }
}

__global__ void per_token_quant_int8_bf16_kernel(int8_t *__restrict__ q_input,
                                                 float *__restrict__ input_scale,
                                                 const __nv_bfloat16 *__restrict__ input,
                                                 const float *__restrict__ smooth_scale,
                                                 int64_t m,
                                                 int64_t k) {
    const int64_t row = blockIdx.x;
    if (row >= m) {
        return;
    }

    const int64_t base = row * k;
    float absmax = 0.0f;
    for (int64_t col = threadIdx.x; col < k; col += blockDim.x) {
        const float value = __bfloat162float(input[base + col]) * smooth_scale[col];
        absmax = fmaxf(absmax, fabsf(value));
    }

    __shared__ float reduce[kThreads];
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
        input_scale[row] = row_scale;
    }

    for (int64_t col = threadIdx.x; col < k; col += blockDim.x) {
        const float value = __bfloat162float(input[base + col]) * smooth_scale[col];
        float q = nearbyintf(value * inv_scale);
        q = fminf(127.0f, fmaxf(-128.0f, q));
        q_input[base + col] = static_cast<int8_t>(q);
    }
}

__global__ void apply_weight_scale_kernel(__nv_bfloat16 *__restrict__ output,
                                          const float *__restrict__ weight_scale,
                                          int64_t m,
                                          int64_t n) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = m * n;
    if (idx >= total) {
        return;
    }
    const int64_t col = idx % n;
    const float value = __bfloat162float(output[idx]) * weight_scale[col];
    output[idx] = __float2bfloat16(value);
}

int grid_for(int64_t total) {
    return static_cast<int>((total + kThreads - 1) / kThreads);
}

} // namespace

void launch_prepare_per_channel_scales(float *input_block_scale,
                                       float *weight_block_scale,
                                       const float *input_scale,
                                       int64_t m,
                                       int64_t n,
                                       int64_t k,
                                       void *stream) {
    const int64_t k_blocks = (k + kBlockSize - 1) / kBlockSize;
    const int64_t n_blocks = (n + kBlockSize - 1) / kBlockSize;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const int64_t input_total = k_blocks * m;
    if (input_total > 0) {
        expand_input_scale_kernel<<<grid_for(input_total), kThreads, 0, cuda_stream>>>(
            input_block_scale, input_scale, m, k_blocks);
    }
    const int64_t weight_total = n_blocks * k_blocks;
    if (weight_total > 0) {
        fill_weight_block_scale_kernel<<<grid_for(weight_total), kThreads, 0, cuda_stream>>>(
            weight_block_scale, weight_total);
    }
}

void launch_per_token_quant_int8_bf16(int8_t *q_input,
                                      float *input_scale,
                                      const void *input,
                                      const float *smooth_scale,
                                      int64_t m,
                                      int64_t k,
                                      void *stream) {
    if (m <= 0 || k <= 0) {
        return;
    }
    per_token_quant_int8_bf16_kernel<<<static_cast<unsigned int>(m),
                                        kThreads,
                                        0,
                                        reinterpret_cast<cudaStream_t>(stream)>>>(
        q_input,
        input_scale,
        reinterpret_cast<const __nv_bfloat16 *>(input),
        smooth_scale,
        m,
        k);
}

void launch_apply_per_channel_weight_scale(void *output,
                                           const float *weight_scale,
                                           int64_t m,
                                           int64_t n,
                                           void *stream) {
    const int64_t total = m * n;
    if (total <= 0) {
        return;
    }
    apply_weight_scale_kernel<<<grid_for(total), kThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<__nv_bfloat16 *>(output),
        weight_scale,
        m,
        n);
}

} // namespace infinicore::op::deepseek_v4_lightop_linear_w8a8_asm_impl
