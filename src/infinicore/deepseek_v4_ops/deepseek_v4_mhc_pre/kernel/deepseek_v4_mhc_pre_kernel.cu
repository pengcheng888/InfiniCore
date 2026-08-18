#include "deepseek_v4_mhc_pre_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace infinicore::op::deepseek_v4_mhc_pre {
namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxHc = 16;

__device__ __forceinline__ float sigmoidf_stable(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void mhc_pre_mix_sqsum_kernel(const __nv_bfloat16 *__restrict__ x,
                                         const float *__restrict__ fn,
                                         float *__restrict__ mixes,
                                         float *__restrict__ sqsum,
                                         int64_t tokens,
                                         int64_t hc,
                                         int64_t hidden,
                                         int64_t mix_hc,
                                         int64_t k_size) {
    const int64_t token = blockIdx.x;
    const int64_t mix = blockIdx.y;
    float dot = 0.0f;
    float ss = 0.0f;
    for (int64_t k = threadIdx.x; k < k_size; k += blockDim.x) {
        const float xv = __bfloat162float(x[token * k_size + k]);
        dot += xv * fn[mix * k_size + k];
        if (mix == 0) {
            ss += xv * xv;
        }
    }
    __shared__ float reduce[kBlockSize];
    reduce[threadIdx.x] = dot;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        mixes[token * mix_hc + mix] = reduce[0];
    }

    if (mix == 0) {
        reduce[threadIdx.x] = ss;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                reduce[threadIdx.x] += reduce[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            sqsum[token] = reduce[0];
        }
    }
}

__global__ void mhc_pre_mix_sqsum_hc4_hidden4096_kernel(const __nv_bfloat16 *__restrict__ x,
                                                        const float *__restrict__ fn,
                                                        float *__restrict__ mixes,
                                                        float *__restrict__ sqsum) {
    constexpr int kSize = 4 * 4096;
    constexpr int kMix = 24;
    const int64_t token = blockIdx.x;
    float acc[kMix];
#pragma unroll
    for (int m = 0; m < kMix; ++m) {
        acc[m] = 0.0f;
    }
    float ss = 0.0f;

    for (int k = threadIdx.x; k < kSize; k += blockDim.x) {
        const float xv = __bfloat162float(x[token * kSize + k]);
        ss += xv * xv;
#pragma unroll
        for (int m = 0; m < kMix; ++m) {
            acc[m] += xv * fn[m * kSize + k];
        }
    }

    __shared__ float reduce[kMix][kBlockSize];
#pragma unroll
    for (int m = 0; m < kMix; ++m) {
        reduce[m][threadIdx.x] = acc[m];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (int m = 0; m < kMix; ++m) {
                reduce[m][threadIdx.x] += reduce[m][threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int m = 0; m < kMix; ++m) {
            mixes[token * kMix + m] = reduce[m][0];
        }
    }

    __shared__ float ss_reduce[kBlockSize];
    ss_reduce[threadIdx.x] = ss;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            ss_reduce[threadIdx.x] += ss_reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        sqsum[token] = ss_reduce[0];
    }
}

__global__ void mhc_pre_mix_sqsum_hc4_hidden4096_group8_kernel(const __nv_bfloat16 *__restrict__ x,
                                                               const float *__restrict__ fn,
                                                               float *__restrict__ mixes,
                                                               float *__restrict__ sqsum) {
    constexpr int kSize = 4 * 4096;
    constexpr int kMix = 24;
    constexpr int kGroup = 8;
    const int64_t token = blockIdx.x;
    const int mix_base = blockIdx.y * kGroup;
    float acc[kGroup];
#pragma unroll
    for (int m = 0; m < kGroup; ++m) {
        acc[m] = 0.0f;
    }
    float ss = 0.0f;

    for (int k = threadIdx.x; k < kSize; k += blockDim.x) {
        const float xv = __bfloat162float(x[token * kSize + k]);
        if (blockIdx.y == 0) {
            ss += xv * xv;
        }
#pragma unroll
        for (int m = 0; m < kGroup; ++m) {
            const int mix = mix_base + m;
            acc[m] += xv * fn[mix * kSize + k];
        }
    }

    __shared__ float reduce[kGroup][kBlockSize];
#pragma unroll
    for (int m = 0; m < kGroup; ++m) {
        reduce[m][threadIdx.x] = acc[m];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (int m = 0; m < kGroup; ++m) {
                reduce[m][threadIdx.x] += reduce[m][threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int m = 0; m < kGroup; ++m) {
            mixes[token * kMix + mix_base + m] = reduce[m][0];
        }
    }

    if (blockIdx.y == 0) {
        __shared__ float ss_reduce[kBlockSize];
        ss_reduce[threadIdx.x] = ss;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                ss_reduce[threadIdx.x] += ss_reduce[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            sqsum[token] = ss_reduce[0];
        }
    }
}

__global__ void mhc_pre_finalize_kernel(float *__restrict__ post,
                                        float *__restrict__ comb,
                                        float *__restrict__ pre,
                                        const float *__restrict__ mixes,
                                        const float *__restrict__ sqsum,
                                        const float *__restrict__ hc_scale,
                                        const float *__restrict__ hc_base,
                                        int64_t hc,
                                        int64_t hidden,
                                        int64_t mix_hc,
                                        double rms_eps,
                                        double hc_pre_eps,
                                        double hc_sinkhorn_eps,
                                        int sinkhorn_repeat) {
    const int64_t token = blockIdx.x;
    if (threadIdx.x != 0 || hc > kMaxHc) {
        return;
    }
    const float rms = rsqrtf(sqsum[token] / static_cast<float>(hc * hidden) + static_cast<float>(rms_eps));
    float local[kMaxHc][kMaxHc];

    for (int64_t h = 0; h < hc; ++h) {
        const float pre_mix = mixes[token * mix_hc + h] * rms;
        const float post_mix = mixes[token * mix_hc + hc + h] * rms;
        pre[token * hc + h] = sigmoidf_stable(pre_mix * hc_scale[0] + hc_base[h]) + static_cast<float>(hc_pre_eps);
        post[token * hc + h] = 2.0f * sigmoidf_stable(post_mix * hc_scale[1] + hc_base[hc + h]);
    }

    for (int64_t hci = 0; hci < hc; ++hci) {
        float row_max = -INFINITY;
        for (int64_t hco = 0; hco < hc; ++hco) {
            const int64_t idx = 2 * hc + hci * hc + hco;
            const float v = mixes[token * mix_hc + idx] * rms * hc_scale[2] + hc_base[idx];
            local[hci][hco] = v;
            row_max = fmaxf(row_max, v);
        }
        float row_sum = 0.0f;
        for (int64_t hco = 0; hco < hc; ++hco) {
            const float v = expf(local[hci][hco] - row_max);
            local[hci][hco] = v;
            row_sum += v;
        }
        for (int64_t hco = 0; hco < hc; ++hco) {
            local[hci][hco] = local[hci][hco] / row_sum + static_cast<float>(hc_sinkhorn_eps);
        }
    }

    for (int64_t hco = 0; hco < hc; ++hco) {
        float col_sum = 0.0f;
        for (int64_t hci = 0; hci < hc; ++hci) {
            col_sum += local[hci][hco];
        }
        const float denom = col_sum + static_cast<float>(hc_sinkhorn_eps);
        for (int64_t hci = 0; hci < hc; ++hci) {
            local[hci][hco] /= denom;
        }
    }

    for (int iter = 1; iter < sinkhorn_repeat; ++iter) {
        for (int64_t hci = 0; hci < hc; ++hci) {
            float row_sum = 0.0f;
            for (int64_t hco = 0; hco < hc; ++hco) {
                row_sum += local[hci][hco];
            }
            const float denom = row_sum + static_cast<float>(hc_sinkhorn_eps);
            for (int64_t hco = 0; hco < hc; ++hco) {
                local[hci][hco] /= denom;
            }
        }
        for (int64_t hco = 0; hco < hc; ++hco) {
            float col_sum = 0.0f;
            for (int64_t hci = 0; hci < hc; ++hci) {
                col_sum += local[hci][hco];
            }
            const float denom = col_sum + static_cast<float>(hc_sinkhorn_eps);
            for (int64_t hci = 0; hci < hc; ++hci) {
                local[hci][hco] /= denom;
            }
        }
    }

    for (int64_t hci = 0; hci < hc; ++hci) {
        for (int64_t hco = 0; hco < hc; ++hco) {
            comb[token * hc * hc + hci * hc + hco] = local[hci][hco];
        }
    }
}

__global__ void mhc_pre_finalize_hc4_hidden4096_kernel(float *__restrict__ post,
                                                       float *__restrict__ comb,
                                                       float *__restrict__ pre,
                                                       const float *__restrict__ mixes,
                                                       const float *__restrict__ sqsum,
                                                       const float *__restrict__ hc_scale,
                                                       const float *__restrict__ hc_base,
                                                       double rms_eps,
                                                       double hc_pre_eps,
                                                       double hc_sinkhorn_eps,
                                                       int sinkhorn_repeat) {
    constexpr int hc = 4;
    constexpr int hidden = 4096;
    constexpr int mix_hc = 24;
    const int64_t token = blockIdx.x;
    if (threadIdx.x != 0) {
        return;
    }

    const float rms = rsqrtf(sqsum[token] / static_cast<float>(hc * hidden) + static_cast<float>(rms_eps));
    const int64_t token_mix = token * mix_hc;
    const int64_t token_hc = token * hc;
    float local[hc][hc];

#pragma unroll
    for (int h = 0; h < hc; ++h) {
        const float pre_mix = mixes[token_mix + h] * rms;
        const float post_mix = mixes[token_mix + hc + h] * rms;
        pre[token_hc + h] = sigmoidf_stable(pre_mix * hc_scale[0] + hc_base[h]) + static_cast<float>(hc_pre_eps);
        post[token_hc + h] = 2.0f * sigmoidf_stable(post_mix * hc_scale[1] + hc_base[hc + h]);
    }

#pragma unroll
    for (int hci = 0; hci < hc; ++hci) {
        float row_max = -INFINITY;
#pragma unroll
        for (int hco = 0; hco < hc; ++hco) {
            const int idx = 2 * hc + hci * hc + hco;
            const float v = mixes[token_mix + idx] * rms * hc_scale[2] + hc_base[idx];
            local[hci][hco] = v;
            row_max = fmaxf(row_max, v);
        }
        float row_sum = 0.0f;
#pragma unroll
        for (int hco = 0; hco < hc; ++hco) {
            const float v = expf(local[hci][hco] - row_max);
            local[hci][hco] = v;
            row_sum += v;
        }
#pragma unroll
        for (int hco = 0; hco < hc; ++hco) {
            local[hci][hco] = local[hci][hco] / row_sum + static_cast<float>(hc_sinkhorn_eps);
        }
    }

#pragma unroll
    for (int hco = 0; hco < hc; ++hco) {
        float col_sum = 0.0f;
#pragma unroll
        for (int hci = 0; hci < hc; ++hci) {
            col_sum += local[hci][hco];
        }
        const float denom = col_sum + static_cast<float>(hc_sinkhorn_eps);
#pragma unroll
        for (int hci = 0; hci < hc; ++hci) {
            local[hci][hco] /= denom;
        }
    }

    for (int iter = 1; iter < sinkhorn_repeat; ++iter) {
#pragma unroll
        for (int hci = 0; hci < hc; ++hci) {
            float row_sum = 0.0f;
#pragma unroll
            for (int hco = 0; hco < hc; ++hco) {
                row_sum += local[hci][hco];
            }
            const float denom = row_sum + static_cast<float>(hc_sinkhorn_eps);
#pragma unroll
            for (int hco = 0; hco < hc; ++hco) {
                local[hci][hco] /= denom;
            }
        }
#pragma unroll
        for (int hco = 0; hco < hc; ++hco) {
            float col_sum = 0.0f;
#pragma unroll
            for (int hci = 0; hci < hc; ++hci) {
                col_sum += local[hci][hco];
            }
            const float denom = col_sum + static_cast<float>(hc_sinkhorn_eps);
#pragma unroll
            for (int hci = 0; hci < hc; ++hci) {
                local[hci][hco] /= denom;
            }
        }
    }

#pragma unroll
    for (int hci = 0; hci < hc; ++hci) {
#pragma unroll
        for (int hco = 0; hco < hc; ++hco) {
            comb[token * hc * hc + hci * hc + hco] = local[hci][hco];
        }
    }
}

__global__ void mhc_pre_y_kernel(__nv_bfloat16 *__restrict__ y,
                                 const __nv_bfloat16 *__restrict__ x,
                                 const float *__restrict__ pre,
                                 int64_t tokens,
                                 int64_t hc,
                                 int64_t hidden) {
    const int64_t token = blockIdx.x;
    const int64_t h = blockIdx.y * blockDim.x + threadIdx.x;
    if (token >= tokens || h >= hidden) {
        return;
    }
    float acc = 0.0f;
    for (int64_t hci = 0; hci < hc; ++hci) {
        acc += pre[token * hc + hci] * __bfloat162float(x[(token * hc + hci) * hidden + h]);
    }
    y[token * hidden + h] = __float2bfloat16(acc);
}

__global__ void mhc_pre_y_hc4_hidden4096_kernel(__nv_bfloat16 *__restrict__ y,
                                                const __nv_bfloat16 *__restrict__ x,
                                                const float *__restrict__ pre) {
    constexpr int hidden = 4096;
    constexpr int hc = 4;
    const int64_t token = blockIdx.x;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= hidden) {
        return;
    }
    const int64_t token_hc = token * hc;
    float acc = 0.0f;
    acc += pre[token_hc + 0] * __bfloat162float(x[(token_hc + 0) * hidden + h]);
    acc += pre[token_hc + 1] * __bfloat162float(x[(token_hc + 1) * hidden + h]);
    acc += pre[token_hc + 2] * __bfloat162float(x[(token_hc + 2) * hidden + h]);
    acc += pre[token_hc + 3] * __bfloat162float(x[(token_hc + 3) * hidden + h]);
    y[token * hidden + h] = __float2bfloat16(acc);
}

} // namespace

void launch_kernel(void *y,
                   float *post,
                   float *comb,
                   const void *residual,
                   const float *fn,
                   const float *hc_scale,
                   const float *hc_base,
                   float *mixes,
                   float *sqsum,
                   float *pre,
                   int64_t tokens,
                   int64_t hc,
                   int64_t hidden,
                   double rms_eps,
                   double hc_pre_eps,
                   double hc_sinkhorn_eps,
                   int sinkhorn_repeat,
                   void *stream) {
    const int64_t mix_hc = (2 + hc) * hc;
    const int64_t k_size = hc * hidden;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    if (hc == 4 && hidden == 4096 && tokens >= 64) {
        dim3 mix_grid(tokens, 3);
        mhc_pre_mix_sqsum_hc4_hidden4096_group8_kernel<<<mix_grid, kBlockSize, 0, cuda_stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(residual), fn, mixes, sqsum);
    } else {
        dim3 mix_grid(tokens, mix_hc);
        mhc_pre_mix_sqsum_kernel<<<mix_grid, kBlockSize, 0, cuda_stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(residual), fn, mixes, sqsum, tokens, hc, hidden, mix_hc, k_size);
    }

    if (hc == 4 && hidden == 4096) {
        mhc_pre_finalize_hc4_hidden4096_kernel<<<tokens, 1, 0, cuda_stream>>>(
            post, comb, pre, mixes, sqsum, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, sinkhorn_repeat);
    } else {
        mhc_pre_finalize_kernel<<<tokens, 1, 0, cuda_stream>>>(
            post, comb, pre, mixes, sqsum, hc_scale, hc_base, hc, hidden, mix_hc, rms_eps, hc_pre_eps, hc_sinkhorn_eps, sinkhorn_repeat);
    }

    dim3 y_grid(tokens, (hidden + kBlockSize - 1) / kBlockSize);
    if (hc == 4 && hidden == 4096) {
        mhc_pre_y_hc4_hidden4096_kernel<<<y_grid, kBlockSize, 0, cuda_stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(y), reinterpret_cast<const __nv_bfloat16 *>(residual), pre);
    } else {
        mhc_pre_y_kernel<<<y_grid, kBlockSize, 0, cuda_stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(y), reinterpret_cast<const __nv_bfloat16 *>(residual), pre, tokens, hc, hidden);
    }
}

} // namespace infinicore::op::deepseek_v4_mhc_pre
