#include "deepseek_v4_hc_head_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace infinicore::op::deepseek_v4_hc_head {
namespace {

constexpr int kBlockSize = 256;

__device__ __forceinline__ float sigmoidf_stable(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void hc_head_mix_sqsum_kernel(const __nv_bfloat16 *__restrict__ x,
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

__global__ void hc_head_mix_sqsum_hc4_hidden4096_kernel(const __nv_bfloat16 *__restrict__ x,
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

__global__ void hc_head_y_kernel(__nv_bfloat16 *__restrict__ y,
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
        const float pre = sigmoidf_stable(mixes[token * hc + hci] * rms * scale[0] + base[hci]) + static_cast<float>(hc_eps);
        acc += pre * __bfloat162float(x[(token * hc + hci) * hidden + h]);
    }
    y[token * hidden + h] = __float2bfloat16(acc);
}

__global__ void hc_head_y_hc4_hidden4096_kernel(__nv_bfloat16 *__restrict__ y,
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

void launch_kernel(void *y,
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
        hc_head_mix_sqsum_hc4_hidden4096_kernel<<<tokens, kBlockSize, 0, cuda_stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(x), fn, mixes, sqsum);
        dim3 y_grid(tokens, (hidden + kBlockSize - 1) / kBlockSize);
        hc_head_y_hc4_hidden4096_kernel<<<y_grid, kBlockSize, 0, cuda_stream>>>(
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
        hc_head_mix_sqsum_kernel<<<mix_grid, kBlockSize, 0, cuda_stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(x), fn, mixes, sqsum, hc, hidden, k_size);
        dim3 y_grid(tokens, (hidden + kBlockSize - 1) / kBlockSize);
        hc_head_y_kernel<<<y_grid, kBlockSize, 0, cuda_stream>>>(
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

} // namespace infinicore::op::deepseek_v4_hc_head
