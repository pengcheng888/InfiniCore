#include "deepseek_v4_mhc_fused_post_pre_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdexcept>

namespace infinicore::op::deepseek_v4_mhc_fused_post_pre {
namespace {

constexpr int kBlockSize = 256;
constexpr int kHc4 = 4;
constexpr int kHidden4096 = 4096;
constexpr int kMixHc4 = 24;
constexpr int kBigFusedTokenThreshold = 128;
constexpr int kWarpThreads = 32;
constexpr int kNumWarps = kBlockSize / kWarpThreads;
constexpr unsigned int kFullWarpMask = 0xffffffffu;

__device__ __forceinline__ float sigmoidf_stable(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = kWarpThreads / 2; offset > 0; offset >>= 1) {
#if defined(__HIP_PLATFORM_AMD__)
        value += __shfl_down(value, offset, kWarpThreads);
#else
        value += __shfl_down_sync(kFullWarpMask, value, offset, kWarpThreads);
#endif
    }
#if defined(__HIP_PLATFORM_AMD__)
    return __shfl(value, 0, kWarpThreads);
#else
    return __shfl_sync(kFullWarpMask, value, 0, kWarpThreads);
#endif
}

__device__ __forceinline__ void reduce_block_sum(float *shared, float value) {
    const int lane = threadIdx.x % kWarpThreads;
    const int warp_id = threadIdx.x / kWarpThreads;
    value = warp_sum(value);
    if (lane == 0) {
        shared[warp_id] = value;
    }
    __syncthreads();

    if (warp_id == 0) {
        value = threadIdx.x < kNumWarps ? shared[lane] : 0.0f;
        value = warp_sum(value);
        if (lane == 0) {
            shared[0] = value;
        }
    }
    __syncthreads();
    return;
}

__global__ void mhc_fused_post_pre_fma_hc4_hidden4096_mix3_split8_kernel(__nv_bfloat16 *__restrict__ residual_cur,
                                                                         const __nv_bfloat16 *__restrict__ x,
                                                                         const __nv_bfloat16 *__restrict__ residual,
                                                                         const float *__restrict__ post_layer_mix,
                                                                         const float *__restrict__ comb_res_mix,
                                                                         const float *__restrict__ fn,
                                                                         float *__restrict__ mixes_partial,
                                                                         float *__restrict__ sqsum_partial) {
    constexpr int kTileMixOutputs = 3;
    constexpr int kSplitK = 8;
    constexpr int kHiddenPerSplit = kHidden4096 / kSplitK;
    const int64_t token = blockIdx.x;
    const int mix_tile = blockIdx.y;
    const int split = blockIdx.z;
    const int mix_base = mix_tile * kTileMixOutputs;
    float acc[kTileMixOutputs] = {0.0f, 0.0f, 0.0f};
    float ss = 0.0f;

    const int hidden_begin = split * kHiddenPerSplit;
    const int hidden_end = hidden_begin + kHiddenPerSplit;
    const int64_t token_hidden = token * kHidden4096;
    const int64_t token_hc = token * kHc4;
    const int64_t token_comb = token * kHc4 * kHc4;

    for (int h = hidden_begin + threadIdx.x; h < hidden_end; h += blockDim.x) {
        const float hidden_in = __bfloat162float(x[token_hidden + h]);
        float cur[kHc4];
#pragma unroll
        for (int new_route = 0; new_route < kHc4; ++new_route) {
            float value = post_layer_mix[token_hc + new_route] * hidden_in;
#pragma unroll
            for (int old_route = 0; old_route < kHc4; ++old_route) {
                value += comb_res_mix[token_comb + old_route * kHc4 + new_route]
                       * __bfloat162float(residual[(token_hc + old_route) * kHidden4096 + h]);
            }
            const __nv_bfloat16 rounded = __float2bfloat16(value);
            cur[new_route] = __bfloat162float(rounded);
            if (mix_tile == 0) {
                residual_cur[(token_hc + new_route) * kHidden4096 + h] = rounded;
                ss += cur[new_route] * cur[new_route];
            }
        }

#pragma unroll
        for (int local_mix = 0; local_mix < kTileMixOutputs; ++local_mix) {
            const int mix = mix_base + local_mix;
            if (mix < kMixHc4) {
                const int64_t fn_base = mix * kHc4 * kHidden4096 + h;
                float dot = 0.0f;
#pragma unroll
                for (int route = 0; route < kHc4; ++route) {
                    dot += cur[route] * fn[fn_base + route * kHidden4096];
                }
                acc[local_mix] += dot;
            }
        }
    }

    __shared__ float reduce[kTileMixOutputs][kBlockSize];
#pragma unroll
    for (int local_mix = 0; local_mix < kTileMixOutputs; ++local_mix) {
        reduce[local_mix][threadIdx.x] = acc[local_mix];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (int local_mix = 0; local_mix < kTileMixOutputs; ++local_mix) {
                reduce[local_mix][threadIdx.x] += reduce[local_mix][threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int local_mix = 0; local_mix < kTileMixOutputs; ++local_mix) {
            const int mix = mix_base + local_mix;
            if (mix < kMixHc4) {
                mixes_partial[(split * gridDim.x + token) * kMixHc4 + mix] = reduce[local_mix][0];
            }
        }
    }

    if (mix_tile == 0) {
        __shared__ float ss_reduce[kBlockSize];
        reduce_block_sum(ss_reduce, ss);
        if (threadIdx.x == 0) {
            sqsum_partial[split * gridDim.x + token] = ss_reduce[0];
        }
    }
    return;
}

__global__ void mhc_pre_mix_sqsum_hc4_hidden4096_per_mix_kernel(const __nv_bfloat16 *__restrict__ x,
                                                                const float *__restrict__ fn,
                                                                float *__restrict__ mixes,
                                                                float *__restrict__ sqsum) {
    constexpr int kSize = kHc4 * kHidden4096;
    const int64_t token = blockIdx.x;
    const int mix = blockIdx.y;
    float dot = 0.0f;
    float ss = 0.0f;

    for (int k = threadIdx.x; k < kSize; k += blockDim.x) {
        const float xv = __bfloat162float(x[token * kSize + k]);
        dot += xv * fn[mix * kSize + k];
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
        mixes[token * kMixHc4 + mix] = reduce[0];
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
    return;
}

__global__ void mhc_pre_mix_sqsum_hc4_hidden4096_group8_kernel(const __nv_bfloat16 *__restrict__ x,
                                                               const float *__restrict__ fn,
                                                               float *__restrict__ mixes,
                                                               float *__restrict__ sqsum) {
    constexpr int kSize = kHc4 * kHidden4096;
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
            mixes[token * kMixHc4 + mix_base + m] = reduce[m][0];
        }
    }

    if (blockIdx.y == 0) {
        __shared__ float ss_reduce[kBlockSize];
        reduce_block_sum(ss_reduce, ss);
        if (threadIdx.x == 0) {
            sqsum[token] = ss_reduce[0];
        }
    }
    return;
}

__global__ void mhc_pre_mix_sqsum_hc4_hidden4096_all24_kernel(const __nv_bfloat16 *__restrict__ x,
                                                              const float *__restrict__ fn,
                                                              float *__restrict__ mixes,
                                                              float *__restrict__ sqsum) {
    constexpr int kSize = kHc4 * kHidden4096;
    float acc[kMixHc4];
#pragma unroll
    for (int m = 0; m < kMixHc4; ++m) {
        acc[m] = 0.0f;
    }
    float ss = 0.0f;
    const int64_t token = blockIdx.x;

    for (int k = threadIdx.x; k < kSize; k += blockDim.x) {
        const float xv = __bfloat162float(x[token * kSize + k]);
        ss += xv * xv;
#pragma unroll
        for (int mix = 0; mix < kMixHc4; ++mix) {
            acc[mix] += xv * fn[mix * kSize + k];
        }
    }

    __shared__ float reduce[kMixHc4][kBlockSize];
#pragma unroll
    for (int mix = 0; mix < kMixHc4; ++mix) {
        reduce[mix][threadIdx.x] = acc[mix];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (int mix = 0; mix < kMixHc4; ++mix) {
                reduce[mix][threadIdx.x] += reduce[mix][threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < kMixHc4) {
        mixes[token * kMixHc4 + threadIdx.x] = reduce[threadIdx.x][0];
    }

    __shared__ float ss_reduce[kBlockSize];
    reduce_block_sum(ss_reduce, ss);
    if (threadIdx.x == 0) {
        sqsum[token] = ss_reduce[0];
    }
    return;
}

__global__ void mhc_post_mix_sqsum_hc4_hidden4096_all24_kernel(__nv_bfloat16 *__restrict__ residual_cur,
                                                               const __nv_bfloat16 *__restrict__ x,
                                                               const __nv_bfloat16 *__restrict__ residual,
                                                               const float *__restrict__ post_layer_mix,
                                                               const float *__restrict__ comb_res_mix,
                                                               const float *__restrict__ fn,
                                                               float *__restrict__ mixes,
                                                               float *__restrict__ sqsum) {
    float acc[kMixHc4];
#pragma unroll
    for (int mix = 0; mix < kMixHc4; ++mix) {
        acc[mix] = 0.0f;
    }
    float ss = 0.0f;
    const int64_t token = blockIdx.x;
    const int64_t token_hidden = token * kHidden4096;
    const int64_t token_hc = token * kHc4;
    const int64_t token_comb = token * kHc4 * kHc4;

    for (int h = threadIdx.x; h < kHidden4096; h += blockDim.x) {
        const float hidden_in = __bfloat162float(x[token_hidden + h]);
        float cur[kHc4];
#pragma unroll
        for (int new_route = 0; new_route < kHc4; ++new_route) {
            float value = post_layer_mix[token_hc + new_route] * hidden_in;
#pragma unroll
            for (int old_route = 0; old_route < kHc4; ++old_route) {
                value += comb_res_mix[token_comb + old_route * kHc4 + new_route]
                       * __bfloat162float(residual[(token_hc + old_route) * kHidden4096 + h]);
            }
            const __nv_bfloat16 rounded = __float2bfloat16(value);
            cur[new_route] = __bfloat162float(rounded);
            residual_cur[(token_hc + new_route) * kHidden4096 + h] = rounded;
            ss += cur[new_route] * cur[new_route];
        }

#pragma unroll
        for (int mix = 0; mix < kMixHc4; ++mix) {
            const int64_t fn_base = mix * kHc4 * kHidden4096 + h;
            float dot = 0.0f;
#pragma unroll
            for (int route = 0; route < kHc4; ++route) {
                dot += cur[route] * fn[fn_base + route * kHidden4096];
            }
            acc[mix] += dot;
        }
    }

    __shared__ float reduce[kMixHc4][kBlockSize];
#pragma unroll
    for (int mix = 0; mix < kMixHc4; ++mix) {
        reduce[mix][threadIdx.x] = acc[mix];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (int mix = 0; mix < kMixHc4; ++mix) {
                reduce[mix][threadIdx.x] += reduce[mix][threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < kMixHc4) {
        mixes[token * kMixHc4 + threadIdx.x] = reduce[threadIdx.x][0];
    }

    __shared__ float ss_reduce[kBlockSize];
    reduce_block_sum(ss_reduce, ss);
    if (threadIdx.x == 0) {
        sqsum[token] = ss_reduce[0];
    }
    return;
}

__device__ void compute_pre_post_comb_hc4(float *__restrict__ post,
                                          float *__restrict__ comb,
                                          float *pre_shared,
                                          const float *__restrict__ mixes_partial,
                                          const float *__restrict__ sqsum_partial,
                                          const float *__restrict__ hc_scale,
                                          const float *__restrict__ hc_base,
                                          int64_t token,
                                          int64_t tokens,
                                          int split_k,
                                          double rms_eps,
                                          double hc_pre_eps,
                                          double hc_sinkhorn_eps,
                                          int sinkhorn_repeat) {
    float mixes[kMixHc4];
#pragma unroll
    for (int m = 0; m < kMixHc4; ++m) {
        mixes[m] = 0.0f;
    }
    float sqsum = 0.0f;
    for (int split = 0; split < split_k; ++split) {
        const int64_t split_token = split * tokens + token;
        sqsum += sqsum_partial[split_token];
#pragma unroll
        for (int m = 0; m < kMixHc4; ++m) {
            mixes[m] += mixes_partial[split_token * kMixHc4 + m];
        }
    }
    const float rms = rsqrtf(sqsum / static_cast<float>(kHc4 * kHidden4096) + static_cast<float>(rms_eps));
    const int64_t token_hc = token * kHc4;
    float local[kHc4][kHc4];

#pragma unroll
    for (int h = 0; h < kHc4; ++h) {
        const float pre_mix = mixes[h] * rms;
        const float post_mix = mixes[kHc4 + h] * rms;
        pre_shared[h] = sigmoidf_stable(pre_mix * hc_scale[0] + hc_base[h]) + static_cast<float>(hc_pre_eps);
        post[token_hc + h] = 2.0f * sigmoidf_stable(post_mix * hc_scale[1] + hc_base[kHc4 + h]);
    }

#pragma unroll
    for (int hci = 0; hci < kHc4; ++hci) {
        float row_max = -INFINITY;
#pragma unroll
        for (int hco = 0; hco < kHc4; ++hco) {
            const int idx = 2 * kHc4 + hci * kHc4 + hco;
            const float v = mixes[idx] * rms * hc_scale[2] + hc_base[idx];
            local[hci][hco] = v;
            row_max = fmaxf(row_max, v);
        }
        float row_sum = 0.0f;
#pragma unroll
        for (int hco = 0; hco < kHc4; ++hco) {
            const float v = expf(local[hci][hco] - row_max);
            local[hci][hco] = v;
            row_sum += v;
        }
#pragma unroll
        for (int hco = 0; hco < kHc4; ++hco) {
            local[hci][hco] = local[hci][hco] / row_sum + static_cast<float>(hc_sinkhorn_eps);
        }
    }

#pragma unroll
    for (int hco = 0; hco < kHc4; ++hco) {
        float col_sum = 0.0f;
#pragma unroll
        for (int hci = 0; hci < kHc4; ++hci) {
            col_sum += local[hci][hco];
        }
        const float denom = col_sum + static_cast<float>(hc_sinkhorn_eps);
#pragma unroll
        for (int hci = 0; hci < kHc4; ++hci) {
            local[hci][hco] /= denom;
        }
    }

    for (int iter = 1; iter < sinkhorn_repeat; ++iter) {
#pragma unroll
        for (int hci = 0; hci < kHc4; ++hci) {
            float row_sum = 0.0f;
#pragma unroll
            for (int hco = 0; hco < kHc4; ++hco) {
                row_sum += local[hci][hco];
            }
            const float denom = row_sum + static_cast<float>(hc_sinkhorn_eps);
#pragma unroll
            for (int hco = 0; hco < kHc4; ++hco) {
                local[hci][hco] /= denom;
            }
        }
#pragma unroll
        for (int hco = 0; hco < kHc4; ++hco) {
            float col_sum = 0.0f;
#pragma unroll
            for (int hci = 0; hci < kHc4; ++hci) {
                col_sum += local[hci][hco];
            }
            const float denom = col_sum + static_cast<float>(hc_sinkhorn_eps);
#pragma unroll
            for (int hci = 0; hci < kHc4; ++hci) {
                local[hci][hco] /= denom;
            }
        }
    }

#pragma unroll
    for (int hci = 0; hci < kHc4; ++hci) {
#pragma unroll
        for (int hco = 0; hco < kHc4; ++hco) {
            comb[token * kHc4 * kHc4 + hci * kHc4 + hco] = local[hci][hco];
        }
    }
    return;
}

__global__ void mhc_pre_finalize_y_norm_hc4_hidden4096_kernel(float *__restrict__ post,
                                                              float *__restrict__ comb,
                                                              __nv_bfloat16 *__restrict__ layer_input,
                                                              const __nv_bfloat16 *__restrict__ residual,
                                                              const float *__restrict__ mixes,
                                                              const float *__restrict__ sqsum,
                                                              const float *__restrict__ hc_scale,
                                                              const float *__restrict__ hc_base,
                                                              const __nv_bfloat16 *__restrict__ norm_weight,
                                                              double rms_eps,
                                                              double hc_pre_eps,
                                                              double hc_sinkhorn_eps,
                                                              int sinkhorn_repeat,
                                                              double norm_eps) {
    const int64_t token = blockIdx.x;
    __shared__ float pre[kHc4];
    __shared__ float reduce[kBlockSize];

    if (threadIdx.x == 0) {
        compute_pre_post_comb_hc4(post,
                                  comb,
                                  pre,
                                  mixes,
                                  sqsum,
                                  hc_scale,
                                  hc_base,
                                  token,
                                  gridDim.x,
                                  1,
                                  rms_eps,
                                  hc_pre_eps,
                                  hc_sinkhorn_eps,
                                  sinkhorn_repeat);
    }
    __syncthreads();

    float ss = 0.0f;
    for (int h = threadIdx.x; h < kHidden4096; h += blockDim.x) {
        const int64_t token_hc = token * kHc4;
        float acc = 0.0f;
        acc += pre[0] * __bfloat162float(residual[(token_hc + 0) * kHidden4096 + h]);
        acc += pre[1] * __bfloat162float(residual[(token_hc + 1) * kHidden4096 + h]);
        acc += pre[2] * __bfloat162float(residual[(token_hc + 2) * kHidden4096 + h]);
        acc += pre[3] * __bfloat162float(residual[(token_hc + 3) * kHidden4096 + h]);
        const __nv_bfloat16 rounded_bf16 = __float2bfloat16(acc);
        const float rounded = __bfloat162float(rounded_bf16);
        ss += rounded * rounded;
        layer_input[token * kHidden4096 + h] = rounded_bf16;
    }

    reduce_block_sum(reduce, ss);
    const float inv = rsqrtf(reduce[0] / static_cast<float>(kHidden4096) + static_cast<float>(norm_eps));

    for (int h = threadIdx.x; h < kHidden4096; h += blockDim.x) {
        const float rounded = __bfloat162float(layer_input[token * kHidden4096 + h]);
        const float w = __bfloat162float(norm_weight[h]);
        layer_input[token * kHidden4096 + h] = __float2bfloat16(rounded * inv * w);
    }
    return;
}

__global__ void mhc_pre_big_fuse_with_norm_hc4_hidden4096_split8_kernel(float *__restrict__ post,
                                                                        float *__restrict__ comb,
                                                                        __nv_bfloat16 *__restrict__ layer_input,
                                                                        const __nv_bfloat16 *__restrict__ residual,
                                                                        const float *__restrict__ mixes_partial,
                                                                        const float *__restrict__ sqsum_partial,
                                                                        const float *__restrict__ hc_scale,
                                                                        const float *__restrict__ hc_base,
                                                                        const __nv_bfloat16 *__restrict__ norm_weight,
                                                                        double rms_eps,
                                                                        double hc_pre_eps,
                                                                        double hc_sinkhorn_eps,
                                                                        int sinkhorn_repeat,
                                                                        double norm_eps) {
    constexpr int kSplitK = 8;
    const int64_t token = blockIdx.x;
    __shared__ float pre[kHc4];
    __shared__ float reduce[kBlockSize];

    if (threadIdx.x == 0) {
        compute_pre_post_comb_hc4(post,
                                  comb,
                                  pre,
                                  mixes_partial,
                                  sqsum_partial,
                                  hc_scale,
                                  hc_base,
                                  token,
                                  gridDim.x,
                                  kSplitK,
                                  rms_eps,
                                  hc_pre_eps,
                                  hc_sinkhorn_eps,
                                  sinkhorn_repeat);
    }
    __syncthreads();

    float ss = 0.0f;
    for (int h = threadIdx.x; h < kHidden4096; h += blockDim.x) {
        const int64_t token_hc = token * kHc4;
        float acc = 0.0f;
        acc += pre[0] * __bfloat162float(residual[(token_hc + 0) * kHidden4096 + h]);
        acc += pre[1] * __bfloat162float(residual[(token_hc + 1) * kHidden4096 + h]);
        acc += pre[2] * __bfloat162float(residual[(token_hc + 2) * kHidden4096 + h]);
        acc += pre[3] * __bfloat162float(residual[(token_hc + 3) * kHidden4096 + h]);
        const __nv_bfloat16 rounded_bf16 = __float2bfloat16(acc);
        const float rounded = __bfloat162float(rounded_bf16);
        ss += rounded * rounded;
        layer_input[token * kHidden4096 + h] = rounded_bf16;
    }

    reduce_block_sum(reduce, ss);
    const float inv = rsqrtf(reduce[0] / static_cast<float>(kHidden4096) + static_cast<float>(norm_eps));

    for (int h = threadIdx.x; h < kHidden4096; h += blockDim.x) {
        const float rounded = __bfloat162float(layer_input[token * kHidden4096 + h]);
        const float w = __bfloat162float(norm_weight[h]);
        layer_input[token * kHidden4096 + h] = __float2bfloat16(rounded * inv * w);
    }
    return;
}

} // namespace

void launch_kernel(void *residual_cur,
                   float *post_mix_cur,
                   float *comb_mix_cur,
                   void *layer_input_cur,
                   const void *x,
                   const void *residual,
                   const float *post_layer_mix,
                   const float *comb_res_mix,
                   const float *fn,
                   const float *hc_scale,
                   const float *hc_base,
                   float *mixes,
                   float *sqsum,
                   float *pre,
                   float *mixes_partial,
                   float *sqsum_partial,
                   int64_t tokens,
                   int64_t hc,
                   int64_t hidden,
                   double rms_eps,
                   double hc_pre_eps,
                   double hc_sinkhorn_eps,
                   double hc_post_mult_value,
                   int sinkhorn_repeat,
                   const void *norm_weight,
                   double norm_eps,
                   void *stream) {
    if (hc_post_mult_value != 2.0) {
        throw std::runtime_error("deepseek_v4_mhc_fused_post_pre kernel currently expects hc_post_mult_value == 2.0.");
    }
    if (tokens == 0) {
        return;
    }

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (hc == kHc4 && hidden == kHidden4096 && tokens < kBigFusedTokenThreshold) {
        constexpr int kTileMixOutputs = 3;
        constexpr int kSplitK = 8;
        dim3 fma_grid(tokens, (kMixHc4 + kTileMixOutputs - 1) / kTileMixOutputs, kSplitK);
        mhc_fused_post_pre_fma_hc4_hidden4096_mix3_split8_kernel<<<fma_grid, kBlockSize, 0, cuda_stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(residual_cur),
            reinterpret_cast<const __nv_bfloat16 *>(x),
            reinterpret_cast<const __nv_bfloat16 *>(residual),
            post_layer_mix,
            comb_res_mix,
            fn,
            mixes_partial,
            sqsum_partial);
        mhc_pre_big_fuse_with_norm_hc4_hidden4096_split8_kernel<<<tokens, kBlockSize, 0, cuda_stream>>>(
            post_mix_cur,
            comb_mix_cur,
            reinterpret_cast<__nv_bfloat16 *>(layer_input_cur),
            reinterpret_cast<const __nv_bfloat16 *>(residual_cur),
            mixes_partial,
            sqsum_partial,
            hc_scale,
            hc_base,
            reinterpret_cast<const __nv_bfloat16 *>(norm_weight),
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            sinkhorn_repeat,
            norm_eps);
        return;
    }

    if (hc == kHc4 && hidden == kHidden4096 && tokens >= kBigFusedTokenThreshold) {
        mhc_post_mix_sqsum_hc4_hidden4096_all24_kernel<<<tokens, kBlockSize, 0, cuda_stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(residual_cur),
            reinterpret_cast<const __nv_bfloat16 *>(x),
            reinterpret_cast<const __nv_bfloat16 *>(residual),
            post_layer_mix,
            comb_res_mix,
            fn,
            mixes,
            sqsum);
        mhc_pre_finalize_y_norm_hc4_hidden4096_kernel<<<tokens, kBlockSize, 0, cuda_stream>>>(
            post_mix_cur,
            comb_mix_cur,
            reinterpret_cast<__nv_bfloat16 *>(layer_input_cur),
            reinterpret_cast<const __nv_bfloat16 *>(residual_cur),
            mixes,
            sqsum,
            hc_scale,
            hc_base,
            reinterpret_cast<const __nv_bfloat16 *>(norm_weight),
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            sinkhorn_repeat,
            norm_eps);
        return;
    }

    throw std::runtime_error("deepseek_v4_mhc_fused_post_pre has no fused kernel for this shape.");
}

} // namespace infinicore::op::deepseek_v4_mhc_fused_post_pre
