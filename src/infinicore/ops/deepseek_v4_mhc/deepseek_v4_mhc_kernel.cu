#include "deepseek_v4_mhc_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace infinicore::op::deepseek_v4_mhc {
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
                                        const float *__restrict__ scale,
                                        const float *__restrict__ base,
                                        int64_t hc,
                                        int64_t hidden,
                                        int64_t mix_hc,
                                        double rms_eps,
                                        double hc_eps,
                                        int sinkhorn_iters) {
    const int64_t token = blockIdx.x;
    if (threadIdx.x != 0 || hc > kMaxHc) {
        return;
    }
    const float rms = rsqrtf(sqsum[token] / static_cast<float>(hc * hidden) + static_cast<float>(rms_eps));
    float local[kMaxHc][kMaxHc];

    for (int64_t h = 0; h < hc; ++h) {
        const float pre_mix = mixes[token * mix_hc + h] * rms;
        const float post_mix = mixes[token * mix_hc + hc + h] * rms;
        pre[token * hc + h] = sigmoidf_stable(pre_mix * scale[0] + base[h]) + static_cast<float>(hc_eps);
        post[token * hc + h] = 2.0f * sigmoidf_stable(post_mix * scale[1] + base[hc + h]);
    }

    for (int64_t hci = 0; hci < hc; ++hci) {
        float row_max = -INFINITY;
        for (int64_t hco = 0; hco < hc; ++hco) {
            const int64_t idx = 2 * hc + hci * hc + hco;
            const float v = mixes[token * mix_hc + idx] * rms * scale[2] + base[idx];
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
            local[hci][hco] = local[hci][hco] / row_sum + static_cast<float>(hc_eps);
        }
    }

    for (int64_t hco = 0; hco < hc; ++hco) {
        float col_sum = 0.0f;
        for (int64_t hci = 0; hci < hc; ++hci) {
            col_sum += local[hci][hco];
        }
        const float denom = col_sum + static_cast<float>(hc_eps);
        for (int64_t hci = 0; hci < hc; ++hci) {
            local[hci][hco] /= denom;
        }
    }

    for (int iter = 1; iter < sinkhorn_iters; ++iter) {
        for (int64_t hci = 0; hci < hc; ++hci) {
            float row_sum = 0.0f;
            for (int64_t hco = 0; hco < hc; ++hco) {
                row_sum += local[hci][hco];
            }
            const float denom = row_sum + static_cast<float>(hc_eps);
            for (int64_t hco = 0; hco < hc; ++hco) {
                local[hci][hco] /= denom;
            }
        }
        for (int64_t hco = 0; hco < hc; ++hco) {
            float col_sum = 0.0f;
            for (int64_t hci = 0; hci < hc; ++hci) {
                col_sum += local[hci][hco];
            }
            const float denom = col_sum + static_cast<float>(hc_eps);
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
                                                       const float *__restrict__ scale,
                                                       const float *__restrict__ base,
                                                       double rms_eps,
                                                       double hc_eps,
                                                       int sinkhorn_iters) {
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
        pre[token_hc + h] = sigmoidf_stable(pre_mix * scale[0] + base[h]) + static_cast<float>(hc_eps);
        post[token_hc + h] = 2.0f * sigmoidf_stable(post_mix * scale[1] + base[hc + h]);
    }

#pragma unroll
    for (int hci = 0; hci < hc; ++hci) {
        float row_max = -INFINITY;
#pragma unroll
        for (int hco = 0; hco < hc; ++hco) {
            const int idx = 2 * hc + hci * hc + hco;
            const float v = mixes[token_mix + idx] * rms * scale[2] + base[idx];
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
            local[hci][hco] = local[hci][hco] / row_sum + static_cast<float>(hc_eps);
        }
    }

#pragma unroll
    for (int hco = 0; hco < hc; ++hco) {
        float col_sum = 0.0f;
#pragma unroll
        for (int hci = 0; hci < hc; ++hci) {
            col_sum += local[hci][hco];
        }
        const float denom = col_sum + static_cast<float>(hc_eps);
#pragma unroll
        for (int hci = 0; hci < hc; ++hci) {
            local[hci][hco] /= denom;
        }
    }

    for (int iter = 1; iter < sinkhorn_iters; ++iter) {
#pragma unroll
        for (int hci = 0; hci < hc; ++hci) {
            float row_sum = 0.0f;
#pragma unroll
            for (int hco = 0; hco < hc; ++hco) {
                row_sum += local[hci][hco];
            }
            const float denom = row_sum + static_cast<float>(hc_eps);
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
            const float denom = col_sum + static_cast<float>(hc_eps);
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

__global__ void mhc_post_kernel(__nv_bfloat16 *__restrict__ y,
                                const __nv_bfloat16 *__restrict__ x,
                                const __nv_bfloat16 *__restrict__ residual,
                                const float *__restrict__ post,
                                const float *__restrict__ comb,
                                int64_t tokens,
                                int64_t hc,
                                int64_t hidden) {
    const int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = tokens * hc * hidden;
    if (flat >= total) {
        return;
    }
    const int64_t h = flat % hidden;
    const int64_t hco = (flat / hidden) % hc;
    const int64_t token = flat / (hidden * hc);
    float acc = post[token * hc + hco] * __bfloat162float(x[token * hidden + h]);
    for (int64_t hci = 0; hci < hc; ++hci) {
        acc += comb[token * hc * hc + hci * hc + hco] *
               __bfloat162float(residual[(token * hc + hci) * hidden + h]);
    }
    y[flat] = __float2bfloat16(acc);
}

__global__ void mhc_post_hc4_hidden4096_kernel(__nv_bfloat16 *__restrict__ y,
                                               const __nv_bfloat16 *__restrict__ x,
                                               const __nv_bfloat16 *__restrict__ residual,
                                               const float *__restrict__ post,
                                               const float *__restrict__ comb) {
    constexpr int hidden = 4096;
    constexpr int hc = 4;
    const int64_t token = blockIdx.z;
    const int hco = blockIdx.y;
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= hidden) {
        return;
    }
    const int64_t token_x = token * hidden + h;
    const int64_t token_hc = token * hc;
    const int64_t comb_base = token * hc * hc + hco;
    float acc = post[token_hc + hco] * __bfloat162float(x[token_x]);
    acc += comb[comb_base + 0 * hc] * __bfloat162float(residual[(token_hc + 0) * hidden + h]);
    acc += comb[comb_base + 1 * hc] * __bfloat162float(residual[(token_hc + 1) * hidden + h]);
    acc += comb[comb_base + 2 * hc] * __bfloat162float(residual[(token_hc + 2) * hidden + h]);
    acc += comb[comb_base + 3 * hc] * __bfloat162float(residual[(token_hc + 3) * hidden + h]);
    y[(token_hc + hco) * hidden + h] = __float2bfloat16(acc);
}

__global__ void mhc_head_mix_sqsum_kernel(const __nv_bfloat16 *__restrict__ x,
                                          const float *__restrict__ fn,
                                          float *__restrict__ mixes,
                                          float *__restrict__ sqsum,
                                          int64_t hc,
                                          int64_t hidden,
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
        mixes[token * hc + mix] = reduce[0];
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

__global__ void mhc_head_mix_sqsum_hc4_hidden4096_kernel(const __nv_bfloat16 *__restrict__ x,
                                                         const float *__restrict__ fn,
                                                         float *__restrict__ mixes,
                                                         float *__restrict__ sqsum) {
    constexpr int kSize = 4 * 4096;
    constexpr int hc = 4;
    const int64_t token = blockIdx.x;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float ss = 0.0f;

    for (int k = threadIdx.x; k < kSize; k += blockDim.x) {
        const float xv = __bfloat162float(x[token * kSize + k]);
        ss += xv * xv;
        acc0 += xv * fn[0 * kSize + k];
        acc1 += xv * fn[1 * kSize + k];
        acc2 += xv * fn[2 * kSize + k];
        acc3 += xv * fn[3 * kSize + k];
    }

    __shared__ float r0[kBlockSize];
    __shared__ float r1[kBlockSize];
    __shared__ float r2[kBlockSize];
    __shared__ float r3[kBlockSize];
    __shared__ float rs[kBlockSize];
    r0[threadIdx.x] = acc0;
    r1[threadIdx.x] = acc1;
    r2[threadIdx.x] = acc2;
    r3[threadIdx.x] = acc3;
    rs[threadIdx.x] = ss;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            r0[threadIdx.x] += r0[threadIdx.x + stride];
            r1[threadIdx.x] += r1[threadIdx.x + stride];
            r2[threadIdx.x] += r2[threadIdx.x + stride];
            r3[threadIdx.x] += r3[threadIdx.x + stride];
            rs[threadIdx.x] += rs[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        mixes[token * hc + 0] = r0[0];
        mixes[token * hc + 1] = r1[0];
        mixes[token * hc + 2] = r2[0];
        mixes[token * hc + 3] = r3[0];
        sqsum[token] = rs[0];
    }
}

__global__ void mhc_head_y_kernel(__nv_bfloat16 *__restrict__ y,
                                  const __nv_bfloat16 *__restrict__ x,
                                  const float *__restrict__ mixes,
                                  const float *__restrict__ sqsum,
                                  const float *__restrict__ scale,
                                  const float *__restrict__ base,
                                  int64_t tokens,
                                  int64_t hc,
                                  int64_t hidden,
                                  double rms_eps,
                                  double hc_eps) {
    const int64_t token = blockIdx.x;
    const int64_t h = blockIdx.y * blockDim.x + threadIdx.x;
    if (token >= tokens || h >= hidden) {
        return;
    }
    const float rms = rsqrtf(sqsum[token] / static_cast<float>(hc * hidden) + static_cast<float>(rms_eps));
    float acc = 0.0f;
    for (int64_t hci = 0; hci < hc; ++hci) {
        const float pre = sigmoidf_stable(mixes[token * hc + hci] * rms * scale[0] + base[hci]) +
                          static_cast<float>(hc_eps);
        acc += pre * __bfloat162float(x[(token * hc + hci) * hidden + h]);
    }
    y[token * hidden + h] = __float2bfloat16(acc);
}

__global__ void mhc_head_y_hc4_hidden4096_kernel(__nv_bfloat16 *__restrict__ y,
                                                 const __nv_bfloat16 *__restrict__ x,
                                                 const float *__restrict__ mixes,
                                                 const float *__restrict__ sqsum,
                                                 const float *__restrict__ scale,
                                                 const float *__restrict__ base,
                                                 double rms_eps,
                                                 double hc_eps) {
    constexpr int hidden = 4096;
    constexpr int hc = 4;
    const int64_t token = blockIdx.x;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= hidden) {
        return;
    }
    const float rms = rsqrtf(sqsum[token] / static_cast<float>(hc * hidden) + static_cast<float>(rms_eps));
    const int64_t token_hc = token * hc;
    const int64_t mix_base = token * hc;
    const float s = scale[0];
    const float pre0 = sigmoidf_stable(mixes[mix_base + 0] * rms * s + base[0]) + static_cast<float>(hc_eps);
    const float pre1 = sigmoidf_stable(mixes[mix_base + 1] * rms * s + base[1]) + static_cast<float>(hc_eps);
    const float pre2 = sigmoidf_stable(mixes[mix_base + 2] * rms * s + base[2]) + static_cast<float>(hc_eps);
    const float pre3 = sigmoidf_stable(mixes[mix_base + 3] * rms * s + base[3]) + static_cast<float>(hc_eps);
    float acc = 0.0f;
    acc += pre0 * __bfloat162float(x[(token_hc + 0) * hidden + h]);
    acc += pre1 * __bfloat162float(x[(token_hc + 1) * hidden + h]);
    acc += pre2 * __bfloat162float(x[(token_hc + 2) * hidden + h]);
    acc += pre3 * __bfloat162float(x[(token_hc + 3) * hidden + h]);
    y[token * hidden + h] = __float2bfloat16(acc);
}

} // namespace

void launch_pre_kernel(void *y,
                       float *post,
                       float *comb,
                       const void *x,
                       const float *fn,
                       const float *scale,
                       const float *base,
                       float *mixes,
                       float *sqsum,
                       float *pre,
                       int64_t tokens,
                       int64_t hc,
                       int64_t hidden,
                       double rms_eps,
                       double hc_eps,
                       int sinkhorn_iters,
                       void *stream) {
    const int64_t mix_hc = (2 + hc) * hc;
    const int64_t k_size = hc * hidden;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (hc == 4 && hidden == 4096 && tokens >= 64) {
        dim3 mix_grid(tokens, 3);
        mhc_pre_mix_sqsum_hc4_hidden4096_group8_kernel<<<mix_grid, kBlockSize, 0, cuda_stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(x), fn, mixes, sqsum);
    } else {
        dim3 mix_grid(tokens, mix_hc);
        mhc_pre_mix_sqsum_kernel<<<mix_grid, kBlockSize, 0, cuda_stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(x), fn, mixes, sqsum, tokens, hc, hidden, mix_hc, k_size);
    }
    if (hc == 4 && hidden == 4096) {
        mhc_pre_finalize_hc4_hidden4096_kernel<<<tokens, 1, 0, cuda_stream>>>(
            post, comb, pre, mixes, sqsum, scale, base, rms_eps, hc_eps, sinkhorn_iters);
    } else {
        mhc_pre_finalize_kernel<<<tokens, 1, 0, cuda_stream>>>(
            post, comb, pre, mixes, sqsum, scale, base, hc, hidden, mix_hc, rms_eps, hc_eps, sinkhorn_iters);
    }
    dim3 y_grid(tokens, (hidden + kBlockSize - 1) / kBlockSize);
    if (hc == 4 && hidden == 4096) {
        mhc_pre_y_hc4_hidden4096_kernel<<<y_grid, kBlockSize, 0, cuda_stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(y), reinterpret_cast<const __nv_bfloat16 *>(x), pre);
    } else {
        mhc_pre_y_kernel<<<y_grid, kBlockSize, 0, cuda_stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(y), reinterpret_cast<const __nv_bfloat16 *>(x), pre, tokens, hc, hidden);
    }
}

void launch_post_kernel(void *y,
                        const void *x,
                        const void *residual,
                        const float *post,
                        const float *comb,
                        int64_t tokens,
                        int64_t hc,
                        int64_t hidden,
                        void *stream) {
    const int64_t total = tokens * hc * hidden;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (hc == 4 && hidden == 4096) {
        dim3 grid((hidden + kBlockSize - 1) / kBlockSize, hc, tokens);
        mhc_post_hc4_hidden4096_kernel<<<grid, kBlockSize, 0, cuda_stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(y),
            reinterpret_cast<const __nv_bfloat16 *>(x),
            reinterpret_cast<const __nv_bfloat16 *>(residual),
            post,
            comb);
    } else {
        mhc_post_kernel<<<(total + kBlockSize - 1) / kBlockSize, kBlockSize, 0, cuda_stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(y),
            reinterpret_cast<const __nv_bfloat16 *>(x),
            reinterpret_cast<const __nv_bfloat16 *>(residual),
            post,
            comb,
            tokens,
            hc,
            hidden);
    }
}

void launch_head_kernel(void *y,
                        const void *x,
                        const float *fn,
                        const float *scale,
                        const float *base,
                        float *mixes,
                        float *sqsum,
                        int64_t tokens,
                        int64_t hc,
                        int64_t hidden,
                        double rms_eps,
                        double hc_eps,
                        void *stream) {
    const int64_t k_size = hc * hidden;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (hc == 4 && hidden == 4096 && tokens >= 128) {
        mhc_head_mix_sqsum_hc4_hidden4096_kernel<<<tokens, kBlockSize, 0, cuda_stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(x), fn, mixes, sqsum);
        dim3 y_grid(tokens, (hidden + kBlockSize - 1) / kBlockSize);
        mhc_head_y_hc4_hidden4096_kernel<<<y_grid, kBlockSize, 0, cuda_stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(y),
            reinterpret_cast<const __nv_bfloat16 *>(x),
            mixes,
            sqsum,
            scale,
            base,
            rms_eps,
            hc_eps);
    } else {
        dim3 mix_grid(tokens, hc);
        mhc_head_mix_sqsum_kernel<<<mix_grid, kBlockSize, 0, cuda_stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(x), fn, mixes, sqsum, hc, hidden, k_size);
        dim3 y_grid(tokens, (hidden + kBlockSize - 1) / kBlockSize);
        mhc_head_y_kernel<<<y_grid, kBlockSize, 0, cuda_stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(y),
            reinterpret_cast<const __nv_bfloat16 *>(x),
            mixes,
            sqsum,
            scale,
            base,
            tokens,
            hc,
            hidden,
            rms_eps,
            hc_eps);
    }
}

} // namespace infinicore::op::deepseek_v4_mhc
