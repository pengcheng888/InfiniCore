#include "deepseek_v4_mhc_pre_kernel_v2.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace infinicore::op::deepseek_v4_mhc_pre_v2 {
namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxHc = 16;
constexpr int kHc4 = 4;
constexpr int kHc4Hidden4096 = 4096;
constexpr int kHc4KSize4096 = kHc4 * kHc4Hidden4096;
constexpr int kHc4Mix = kHc4 * (2 + kHc4);
constexpr int kTiledTokenBlock = 32;
constexpr int kTiledHiddenBlock = 256;
constexpr int kTiledSplitK = 32;
constexpr int kTiledMixStride = 32;
constexpr int kTiledMixGroup = 8;
constexpr int kDsv4SinkhornRepeat = 20;

__device__ __forceinline__ float sigmoidf_stable(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void mhc_pre_splitk_gemm_sqsum_kernel(const __nv_bfloat16 *__restrict__ residual,
                                                 const float *__restrict__ fn,
                                                 float *__restrict__ partial_mixes,
                                                 float *__restrict__ partial_sqsum,
                                                 int64_t tokens,
                                                 int64_t hc,
                                                 int64_t hidden,
                                                 int64_t mix_hc,
                                                 int64_t k_size,
                                                 int split_k) {
    const int64_t token = blockIdx.x;
    const int split = blockIdx.y;
    const int64_t mix = blockIdx.z;
    const int64_t split_size = (k_size + split_k - 1) / split_k;
    const int64_t k_begin = split * split_size;
    const int64_t k_end = min(k_begin + split_size, k_size);

    float dot = 0.0f;
    float ss = 0.0f;
    for (int64_t k = k_begin + threadIdx.x; k < k_end; k += blockDim.x) {
        const float xv = __bfloat162float(residual[token * k_size + k]);
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
        partial_mixes[(static_cast<int64_t>(split) * tokens + token) * mix_hc + mix] = reduce[0];
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
            partial_sqsum[static_cast<int64_t>(split) * tokens + token] = reduce[0];
        }
    }
}

__global__ void mhc_pre_gemm_sqsum_hc4_hidden4096_group8_kernel(const __nv_bfloat16 *__restrict__ residual,
                                                                const float *__restrict__ fn,
                                                                float *__restrict__ partial_mixes,
                                                                float *__restrict__ partial_sqsum) {
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
        const float xv = __bfloat162float(residual[token * kSize + k]);
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
            partial_mixes[token * kMix + mix_base + m] = reduce[m][0];
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
            partial_sqsum[token] = ss_reduce[0];
        }
    }
}

__global__ void mhc_pre_gemm_sqsum_hc4_hidden4096_group12_kernel(const __nv_bfloat16 *__restrict__ residual,
                                                                 const float *__restrict__ fn,
                                                                 float *__restrict__ partial_mixes,
                                                                 float *__restrict__ partial_sqsum) {
    constexpr int kSize = 4 * 4096;
    constexpr int kMix = 24;
    constexpr int kGroup = 12;
    const int64_t token = blockIdx.x;
    const int mix_base = blockIdx.y * kGroup;
    float acc[kGroup];
#pragma unroll
    for (int m = 0; m < kGroup; ++m) {
        acc[m] = 0.0f;
    }
    float ss = 0.0f;

    for (int k = threadIdx.x; k < kSize; k += blockDim.x) {
        const float xv = __bfloat162float(residual[token * kSize + k]);
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
            partial_mixes[token * kMix + mix_base + m] = reduce[m][0];
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
            partial_sqsum[token] = ss_reduce[0];
        }
    }
}

__global__ void mhc_pre_splitk_gemm_sqsum_group8_kernel(const __nv_bfloat16 *__restrict__ residual,
                                                        const float *__restrict__ fn,
                                                        float *__restrict__ partial_mixes,
                                                        float *__restrict__ partial_sqsum,
                                                        int64_t tokens,
                                                        int64_t hidden,
                                                        int64_t mix_hc,
                                                        int64_t k_size,
                                                        int split_k) {
    constexpr int kGroup = 8;
    const int64_t token = blockIdx.x;
    const int split = blockIdx.y;
    const int64_t mix_base = static_cast<int64_t>(blockIdx.z) * kGroup;
    const int64_t split_size = (k_size + split_k - 1) / split_k;
    const int64_t k_begin = split * split_size;
    const int64_t k_end = min(k_begin + split_size, k_size);

    float acc[kGroup];
#pragma unroll
    for (int m = 0; m < kGroup; ++m) {
        acc[m] = 0.0f;
    }
    float ss = 0.0f;

    for (int64_t k = k_begin + threadIdx.x; k < k_end; k += blockDim.x) {
        const float xv = __bfloat162float(residual[token * k_size + k]);
        if (blockIdx.z == 0) {
            ss += xv * xv;
        }
#pragma unroll
        for (int m = 0; m < kGroup; ++m) {
            const int64_t mix = mix_base + m;
            if (mix < mix_hc) {
                acc[m] += xv * fn[mix * k_size + k];
            }
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
            const int64_t mix = mix_base + m;
            if (mix < mix_hc) {
                partial_mixes[(static_cast<int64_t>(split) * tokens + token) * mix_hc + mix] = reduce[m][0];
            }
        }
    }

    if (blockIdx.z == 0) {
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
            partial_sqsum[static_cast<int64_t>(split) * tokens + token] = ss_reduce[0];
        }
    }
}

__global__ void mhc_pre_splitk_tiled_group8_hc4_hidden4096_kernel(const __nv_bfloat16 *__restrict__ residual,
                                                                  const float *__restrict__ fn,
                                                                  float *__restrict__ partial_mixes,
                                                                  float *__restrict__ partial_sqsum,
                                                                  int64_t tokens) {
    constexpr int split_size = kHc4KSize4096 / kTiledSplitK;
    constexpr int tiles_per_split = split_size / kTiledHiddenBlock;
    const int64_t token_base = static_cast<int64_t>(blockIdx.x) * kTiledTokenBlock;
    const int split = blockIdx.y;
    const int mix_base = blockIdx.z * kTiledMixGroup;
    const int tid = threadIdx.x;
    const int local_token = tid / kTiledMixGroup;
    const int local_mix = tid - local_token * kTiledMixGroup;
    const int mix = mix_base + local_mix;

    __shared__ __nv_bfloat16 x_smem[kTiledTokenBlock][kTiledHiddenBlock];
    __shared__ float fn_smem[kTiledMixGroup][kTiledHiddenBlock];
    __shared__ float ss_shared[kTiledTokenBlock][kTiledMixGroup];

    float acc = 0.0f;
    float ss_acc = 0.0f;

    for (int tile = 0; tile < tiles_per_split; ++tile) {
        const int k_tile_base = split * split_size + tile * kTiledHiddenBlock;
        for (int idx = tid; idx < kTiledTokenBlock * kTiledHiddenBlock; idx += kBlockSize) {
            const int t = idx / kTiledHiddenBlock;
            const int k_local = idx - t * kTiledHiddenBlock;
            const int64_t token = token_base + t;
            x_smem[t][k_local] = token < tokens ? residual[token * kHc4KSize4096 + k_tile_base + k_local] : __float2bfloat16(0.0f);
        }
        for (int idx = tid; idx < kTiledMixGroup * kTiledHiddenBlock; idx += kBlockSize) {
            const int m = idx / kTiledHiddenBlock;
            const int k_local = idx - m * kTiledHiddenBlock;
            fn_smem[m][k_local] = fn[(mix_base + m) * kHc4KSize4096 + k_tile_base + k_local];
        }
        __syncthreads();

#pragma unroll 4
        for (int k_local = 0; k_local < kTiledHiddenBlock; ++k_local) {
            const float xv = __bfloat162float(x_smem[local_token][k_local]);
            acc += xv * fn_smem[local_mix][k_local];
        }
        if (blockIdx.z == 0) {
            for (int k_local = local_mix; k_local < kTiledHiddenBlock; k_local += kTiledMixGroup) {
                const float xv = __bfloat162float(x_smem[local_token][k_local]);
                ss_acc += xv * xv;
            }
        }
        __syncthreads();
    }

    const int64_t token = token_base + local_token;
    if (token < tokens) {
        partial_mixes[(static_cast<int64_t>(split) * tokens + token) * kTiledMixStride + mix] = acc;
    }

    if (blockIdx.z == 0) {
        ss_shared[local_token][local_mix] = ss_acc;
        __syncthreads();
        if (local_mix == 0) {
            float ss = 0.0f;
#pragma unroll
            for (int lane = 0; lane < kTiledMixGroup; ++lane) {
                ss += ss_shared[local_token][lane];
            }
            if (token < tokens) {
                partial_sqsum[static_cast<int64_t>(split) * tokens + token] = ss;
            }
        }
    }
}

__global__ void mhc_pre_smallk_hc4_kernel(__nv_bfloat16 *__restrict__ y,
                                          float *__restrict__ post,
                                          float *__restrict__ comb,
                                          const __nv_bfloat16 *__restrict__ residual,
                                          const float *__restrict__ fn,
                                          const float *__restrict__ hc_scale,
                                          const float *__restrict__ hc_base,
                                          int64_t tokens,
                                          int64_t hidden,
                                          int64_t k_size,
                                          double rms_eps,
                                          double hc_pre_eps,
                                          double hc_sinkhorn_eps,
                                          int sinkhorn_repeat) {
    constexpr int hc = 4;
    constexpr int mix_hc = 24;
    const int64_t token = blockIdx.x;
    if (token >= tokens) {
        return;
    }

    float acc[mix_hc];
#pragma unroll
    for (int m = 0; m < mix_hc; ++m) {
        acc[m] = 0.0f;
    }
    float ss = 0.0f;

    for (int64_t k = threadIdx.x; k < k_size; k += blockDim.x) {
        const float xv = __bfloat162float(residual[token * k_size + k]);
        ss += xv * xv;
#pragma unroll
        for (int m = 0; m < mix_hc; ++m) {
            acc[m] += xv * fn[m * k_size + k];
        }
    }

    __shared__ float reduce[mix_hc][kBlockSize];
#pragma unroll
    for (int m = 0; m < mix_hc; ++m) {
        reduce[m][threadIdx.x] = acc[m];
    }
    __shared__ float ss_reduce[kBlockSize];
    ss_reduce[threadIdx.x] = ss;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (int m = 0; m < mix_hc; ++m) {
                reduce[m][threadIdx.x] += reduce[m][threadIdx.x + stride];
            }
            ss_reduce[threadIdx.x] += ss_reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }

    __shared__ float pre_shared[hc];
    if (threadIdx.x == 0) {
        const float rms = rsqrtf(ss_reduce[0] / static_cast<float>(k_size) + static_cast<float>(rms_eps));

#pragma unroll
        for (int h = 0; h < hc; ++h) {
            pre_shared[h] = sigmoidf_stable(reduce[h][0] * rms * hc_scale[0] + hc_base[h]) + static_cast<float>(hc_pre_eps);
            post[token * hc + h] = 2.0f * sigmoidf_stable(reduce[hc + h][0] * rms * hc_scale[1] + hc_base[hc + h]);
        }

        float local[hc][hc];
#pragma unroll
        for (int hci = 0; hci < hc; ++hci) {
            float row_max = -INFINITY;
#pragma unroll
            for (int hco = 0; hco < hc; ++hco) {
                const int idx = 2 * hc + hci * hc + hco;
                const float v = reduce[idx][0] * rms * hc_scale[2] + hc_base[idx];
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
    __syncthreads();

    for (int64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
        float yv = 0.0f;
#pragma unroll
        for (int hci = 0; hci < hc; ++hci) {
            yv += pre_shared[hci] * __bfloat162float(residual[(token * hc + hci) * hidden + h]);
        }
        y[token * hidden + h] = __float2bfloat16(yv);
    }
}

__global__ void mhc_pre_big_fuse_kernel(__nv_bfloat16 *__restrict__ y,
                                        float *__restrict__ post,
                                        float *__restrict__ comb,
                                        const __nv_bfloat16 *__restrict__ residual,
                                        const float *__restrict__ partial_mixes,
                                        const float *__restrict__ partial_sqsum,
                                        const float *__restrict__ hc_scale,
                                        const float *__restrict__ hc_base,
                                        int64_t tokens,
                                        int64_t hc,
                                        int64_t hidden,
                                        int64_t mix_hc,
                                        int64_t k_size,
                                        double rms_eps,
                                        double hc_pre_eps,
                                        double hc_sinkhorn_eps,
                                        int sinkhorn_repeat,
                                        int split_k,
                                        int partial_stride) {
    const int64_t token = blockIdx.x;
    const int64_t hidden_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (token >= tokens || hc > kMaxHc) {
        return;
    }

    __shared__ float pre_shared[kMaxHc];
    if (threadIdx.x == 0) {
        float sqsum = 0.0f;
        for (int split = 0; split < split_k; ++split) {
            sqsum += partial_sqsum[static_cast<int64_t>(split) * tokens + token];
        }
        const float rms = rsqrtf(sqsum / static_cast<float>(hc * hidden) + static_cast<float>(rms_eps));

        for (int64_t h = 0; h < hc; ++h) {
            float pre_mix = 0.0f;
            for (int split = 0; split < split_k; ++split) {
                pre_mix += partial_mixes[(static_cast<int64_t>(split) * tokens + token) * partial_stride + h];
            }
            pre_shared[h] = sigmoidf_stable(pre_mix * rms * hc_scale[0] + hc_base[h]) + static_cast<float>(hc_pre_eps);
        }

        if (blockIdx.y == 0) {
            for (int64_t h = 0; h < hc; ++h) {
                float post_mix = 0.0f;
                for (int split = 0; split < split_k; ++split) {
                    post_mix += partial_mixes[(static_cast<int64_t>(split) * tokens + token) * partial_stride + hc + h];
                }
                post[token * hc + h] = 2.0f * sigmoidf_stable(post_mix * rms * hc_scale[1] + hc_base[hc + h]);
            }

            float local[kMaxHc][kMaxHc];
            for (int64_t hci = 0; hci < hc; ++hci) {
                float row_max = -INFINITY;
                for (int64_t hco = 0; hco < hc; ++hco) {
                    const int64_t idx = 2 * hc + hci * hc + hco;
                    float comb_mix = 0.0f;
                    for (int split = 0; split < split_k; ++split) {
                        comb_mix += partial_mixes[(static_cast<int64_t>(split) * tokens + token) * partial_stride + idx];
                    }
                    const float v = comb_mix * rms * hc_scale[2] + hc_base[idx];
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
    }
    __syncthreads();

    if (hidden_idx < hidden) {
        float acc = 0.0f;
        for (int64_t hci = 0; hci < hc; ++hci) {
            acc += pre_shared[hci] * __bfloat162float(residual[(token * hc + hci) * hidden + hidden_idx]);
        }
        y[token * hidden + hidden_idx] = __float2bfloat16(acc);
    }
}

__global__ void mhc_pre_big_fuse_hc4_kernel(__nv_bfloat16 *__restrict__ y,
                                            float *__restrict__ post,
                                            float *__restrict__ comb,
                                            const __nv_bfloat16 *__restrict__ residual,
                                            const float *__restrict__ partial_mixes,
                                            const float *__restrict__ partial_sqsum,
                                            const float *__restrict__ hc_scale,
                                            const float *__restrict__ hc_base,
                                            int64_t tokens,
                                            int64_t hidden,
                                            int64_t mix_hc,
                                            int64_t k_size,
                                            double rms_eps,
                                            double hc_pre_eps,
                                            double hc_sinkhorn_eps,
                                            int sinkhorn_repeat,
                                            int split_k,
                                            int partial_stride) {
    constexpr int hc = 4;
    const int64_t token = blockIdx.x;
    const int64_t hidden_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (token >= tokens) {
        return;
    }

    __shared__ float pre_shared[hc];
    if (threadIdx.x == 0) {
        float sqsum = 0.0f;
        for (int split = 0; split < split_k; ++split) {
            sqsum += partial_sqsum[static_cast<int64_t>(split) * tokens + token];
        }
        const float rms = rsqrtf(sqsum / static_cast<float>(hc * hidden) + static_cast<float>(rms_eps));

#pragma unroll
        for (int h = 0; h < hc; ++h) {
            float pre_mix = 0.0f;
            for (int split = 0; split < split_k; ++split) {
                pre_mix += partial_mixes[(static_cast<int64_t>(split) * tokens + token) * partial_stride + h];
            }
            pre_shared[h] = sigmoidf_stable(pre_mix * rms * hc_scale[0] + hc_base[h]) + static_cast<float>(hc_pre_eps);
        }

        if (blockIdx.y == 0) {
#pragma unroll
            for (int h = 0; h < hc; ++h) {
                float post_mix = 0.0f;
                for (int split = 0; split < split_k; ++split) {
                    post_mix += partial_mixes[(static_cast<int64_t>(split) * tokens + token) * partial_stride + hc + h];
                }
                post[token * hc + h] = 2.0f * sigmoidf_stable(post_mix * rms * hc_scale[1] + hc_base[hc + h]);
            }

            float local[hc][hc];
#pragma unroll
            for (int hci = 0; hci < hc; ++hci) {
                float row_max = -INFINITY;
#pragma unroll
                for (int hco = 0; hco < hc; ++hco) {
                    const int idx = 2 * hc + hci * hc + hco;
                    float comb_mix = 0.0f;
                    for (int split = 0; split < split_k; ++split) {
                        comb_mix += partial_mixes[(static_cast<int64_t>(split) * tokens + token) * partial_stride + idx];
                    }
                    const float v = comb_mix * rms * hc_scale[2] + hc_base[idx];
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
    }
    __syncthreads();

    if (hidden_idx < hidden) {
        float acc = 0.0f;
#pragma unroll
        for (int hci = 0; hci < hc; ++hci) {
            acc += pre_shared[hci] * __bfloat162float(residual[(token * hc + hci) * hidden + hidden_idx]);
        }
        y[token * hidden + hidden_idx] = __float2bfloat16(acc);
    }
}

} // namespace

void launch_kernel(void *y,
                   float *post,
                   float *comb,
                   const void *residual,
                   const float *fn,
                   const float *hc_scale,
                   const float *hc_base,
                   float *partial_mixes,
                   float *partial_sqsum,
                   int64_t tokens,
                   int64_t hc,
                   int64_t hidden,
                   double rms_eps,
                   double hc_pre_eps,
                   double hc_sinkhorn_eps,
                   int sinkhorn_repeat,
                   int split_k,
                   int partial_stride,
                   void *stream) {
    const int64_t mix_hc = (2 + hc) * hc;
    const int64_t k_size = hc * hidden;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    if (hc == 4 && k_size <= 2048) {
        mhc_pre_smallk_hc4_kernel<<<tokens, kBlockSize, 0, cuda_stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(y),
            post,
            comb,
            reinterpret_cast<const __nv_bfloat16 *>(residual),
            fn,
            hc_scale,
            hc_base,
            tokens,
            hidden,
            k_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            sinkhorn_repeat);
        return;
    }

    if (hc == 4 && hidden == 4096 && split_k == kTiledSplitK && partial_stride == kTiledMixStride) {
        dim3 gemm_grid((tokens + kTiledTokenBlock - 1) / kTiledTokenBlock, kTiledSplitK, kHc4Mix / kTiledMixGroup);
        mhc_pre_splitk_tiled_group8_hc4_hidden4096_kernel<<<gemm_grid, kBlockSize, 0, cuda_stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(residual),
            fn,
            partial_mixes,
            partial_sqsum,
            tokens);
    } else if (hc == 4 && hidden == 4096 && split_k == 1 && sinkhorn_repeat == kDsv4SinkhornRepeat && tokens <= 1024) {
        dim3 gemm_grid(tokens, 2);
        mhc_pre_gemm_sqsum_hc4_hidden4096_group12_kernel<<<gemm_grid, kBlockSize, 0, cuda_stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(residual),
            fn,
            partial_mixes,
            partial_sqsum);
    } else if (hc == 4 && hidden == 4096 && split_k == 1) {
        dim3 gemm_grid(tokens, 3);
        mhc_pre_gemm_sqsum_hc4_hidden4096_group8_kernel<<<gemm_grid, kBlockSize, 0, cuda_stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(residual),
            fn,
            partial_mixes,
            partial_sqsum);
    } else {
        dim3 gemm_grid(tokens, split_k, (mix_hc + 7) / 8);
        mhc_pre_splitk_gemm_sqsum_group8_kernel<<<gemm_grid, kBlockSize, 0, cuda_stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(residual),
            fn,
            partial_mixes,
            partial_sqsum,
            tokens,
            hidden,
            mix_hc,
            k_size,
            split_k);
    }

    dim3 fuse_grid(tokens, (hidden + kBlockSize - 1) / kBlockSize);
    if (hc == 4) {
        mhc_pre_big_fuse_hc4_kernel<<<fuse_grid, kBlockSize, 0, cuda_stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(y),
            post,
            comb,
            reinterpret_cast<const __nv_bfloat16 *>(residual),
            partial_mixes,
            partial_sqsum,
            hc_scale,
            hc_base,
            tokens,
            hidden,
            mix_hc,
            k_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            sinkhorn_repeat,
            split_k,
            partial_stride);
    } else {
        mhc_pre_big_fuse_kernel<<<fuse_grid, kBlockSize, 0, cuda_stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(y),
            post,
            comb,
            reinterpret_cast<const __nv_bfloat16 *>(residual),
            partial_mixes,
            partial_sqsum,
            hc_scale,
            hc_base,
            tokens,
            hc,
            hidden,
            mix_hc,
            k_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            sinkhorn_repeat,
            split_k,
            partial_stride);
    }
}

} // namespace infinicore::op::deepseek_v4_mhc_pre_v2
