#include "deepseek_v4_mhc_post_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace infinicore::op::deepseek_v4_mhc_post {
namespace {

constexpr int kBlockSize = 256;
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
        acc += comb[token * hc * hc + hci * hc + hco] * __bfloat162float(residual[(token * hc + hci) * hidden + h]);
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

} // namespace

void launch_kernel(void *y,
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

} // namespace infinicore::op::deepseek_v4_mhc_post
