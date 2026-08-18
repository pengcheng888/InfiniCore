#include "deepseek_v4_linear_bf16_fp32_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace infinicore::op::deepseek_v4_linear_bf16_fp32_impl {
namespace {

constexpr int kBlockSize = 256;
constexpr int kDsv4Hidden = 4096;
constexpr int kDsv4OutFeatures = 256;
constexpr int kDotKernelMaxTokens = 16;
constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kTileK = 128;
constexpr int kGroupN = 8;

__global__ void linear_bf16_fp32_kernel(float *__restrict__ out,
                                        const __nv_bfloat16 *__restrict__ x,
                                        const __nv_bfloat16 *__restrict__ weight,
                                        int64_t tokens,
                                        int64_t out_features,
                                        int64_t in_features) {
    const int64_t token = blockIdx.x;
    const int64_t out_feature = blockIdx.y;
    if (token >= tokens || out_feature >= out_features) {
        return;
    }

    float sum = 0.0f;
    for (int64_t k = threadIdx.x; k < in_features; k += blockDim.x) {
        const float xv = __bfloat162float(x[token * in_features + k]);
        const float wv = __bfloat162float(weight[out_feature * in_features + k]);
        sum += xv * wv;
    }

    __shared__ float reduce[kBlockSize];
    reduce[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[token * out_features + out_feature] = reduce[0];
    }
}

__global__ void linear_bf16_fp32_dsv4_tiled_kernel(float *__restrict__ out,
                                                   const __nv_bfloat16 *__restrict__ x,
                                                   const __nv_bfloat16 *__restrict__ weight,
                                                   int64_t tokens) {
    const int local_col = threadIdx.x;
    const int local_row = threadIdx.y;
    const int tid = local_row * blockDim.x + local_col;
    const int token = blockIdx.x * kTileM + local_row;
    const int out_feature = blockIdx.y * kTileN + local_col;

    __shared__ __nv_bfloat16 x_tile[kTileM][kTileK];
    __shared__ __nv_bfloat16 w_tile[kTileN][kTileK];

    float sum = 0.0f;
#pragma unroll
    for (int k0 = 0; k0 < kDsv4Hidden; k0 += kTileK) {
        for (int idx = tid; idx < kTileM * kTileK; idx += kTileM * kTileN) {
            const int row = idx / kTileK;
            const int k = idx - row * kTileK;
            const int global_token = blockIdx.x * kTileM + row;
            x_tile[row][k] = global_token < tokens ? x[global_token * kDsv4Hidden + k0 + k] : __float2bfloat16(0.0f);
        }
        for (int idx = tid; idx < kTileN * kTileK; idx += kTileM * kTileN) {
            const int col = idx / kTileK;
            const int k = idx - col * kTileK;
            const int global_out = blockIdx.y * kTileN + col;
            w_tile[col][k] = weight[global_out * kDsv4Hidden + k0 + k];
        }
        __syncthreads();

        if (token < tokens) {
#pragma unroll
            for (int k = 0; k < kTileK; ++k) {
                sum += __bfloat162float(x_tile[local_row][k]) * __bfloat162float(w_tile[local_col][k]);
            }
        }
        __syncthreads();
    }

    if (token < tokens) {
        out[token * kDsv4OutFeatures + out_feature] = sum;
    }
}

__global__ void linear_bf16_fp32_dsv4_out8_kernel(float *__restrict__ out,
                                                  const __nv_bfloat16 *__restrict__ x,
                                                  const __nv_bfloat16 *__restrict__ weight,
                                                  int64_t tokens) {
    const int64_t token = blockIdx.x;
    const int out_base = blockIdx.y * kGroupN;
    const int lane = threadIdx.x;
    if (token >= tokens) {
        return;
    }

    float sum[kGroupN];
#pragma unroll
    for (int i = 0; i < kGroupN; ++i) {
        sum[i] = 0.0f;
    }

    for (int64_t k = lane; k < kDsv4Hidden; k += blockDim.x) {
        const float xv = __bfloat162float(x[token * kDsv4Hidden + k]);
#pragma unroll
        for (int i = 0; i < kGroupN; ++i) {
            const int out_feature = out_base + i;
            const float wv = __bfloat162float(weight[out_feature * kDsv4Hidden + k]);
            sum[i] += xv * wv;
        }
    }

    __shared__ float reduce[kGroupN][kBlockSize];
#pragma unroll
    for (int i = 0; i < kGroupN; ++i) {
        reduce[i][lane] = sum[i];
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
#pragma unroll
            for (int i = 0; i < kGroupN; ++i) {
                reduce[i][lane] += reduce[i][lane + stride];
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < kGroupN; ++i) {
            out[token * kDsv4OutFeatures + out_base + i] = reduce[i][0];
        }
    }
}

} // namespace

void launch_linear_bf16_fp32(float *out,
                             const void *x,
                             const void *weight,
                             int64_t tokens,
                             int64_t out_features,
                             int64_t in_features,
                             void *stream) {
    if (tokens <= 0 || out_features <= 0 || in_features <= 0) {
        return;
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (in_features == kDsv4Hidden && out_features == kDsv4OutFeatures && tokens > kDotKernelMaxTokens) {
        dim3 grid(static_cast<unsigned int>(tokens), static_cast<unsigned int>(kDsv4OutFeatures / kGroupN));
        linear_bf16_fp32_dsv4_out8_kernel<<<grid, kBlockSize, 0, cuda_stream>>>(
            out,
            reinterpret_cast<const __nv_bfloat16 *>(x),
            reinterpret_cast<const __nv_bfloat16 *>(weight),
            tokens);
        return;
    }

    dim3 grid(static_cast<unsigned int>(tokens), static_cast<unsigned int>(out_features));
    linear_bf16_fp32_kernel<<<grid, kBlockSize, 0, cuda_stream>>>(
        out,
        reinterpret_cast<const __nv_bfloat16 *>(x),
        reinterpret_cast<const __nv_bfloat16 *>(weight),
        tokens,
        out_features,
        in_features);
}

} // namespace infinicore::op::deepseek_v4_linear_bf16_fp32_impl
