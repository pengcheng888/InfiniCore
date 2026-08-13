#include "deepseek_v4_compress_fused_norm_rope_kernel.hpp"

#include "../deepseek_v4_compress_common/deepseek_v4_compress_dtype.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace infinicore::op::deepseek_v4_compress_fused_norm_rope_kernel {
namespace {

__device__ __forceinline__ float load_scalar(const void *__restrict__ ptr, int64_t idx, int dtype) {
    using namespace infinicore::op::deepseek_v4_compress_common;
    if (dtype == kDsv4BF16) {
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(ptr)[idx]);
    }
    if (dtype == kDsv4F16) {
        return __half2float(reinterpret_cast<const __half *>(ptr)[idx]);
    }
    return reinterpret_cast<const float *>(ptr)[idx];
}

__device__ __forceinline__ void store_scalar(void *__restrict__ ptr, int64_t idx, int dtype, float value) {
    using namespace infinicore::op::deepseek_v4_compress_common;
    if (dtype == kDsv4BF16) {
        reinterpret_cast<__nv_bfloat16 *>(ptr)[idx] = __float2bfloat16(value);
    } else if (dtype == kDsv4F16) {
        reinterpret_cast<__half *>(ptr)[idx] = __float2half(value);
    } else {
        reinterpret_cast<float *>(ptr)[idx] = value;
    }
}

__device__ __forceinline__ int64_t load_index(const void *__restrict__ ptr, int64_t idx, bool i64) {
    return i64 ? reinterpret_cast<const int64_t *>(ptr)[idx]
               : static_cast<int64_t>(reinterpret_cast<const int32_t *>(ptr)[idx]);
}

__global__ void compress_fused_norm_rope_kernel(void *__restrict__ input,
                                                int input_dtype,
                                                const void *__restrict__ norm_weight,
                                                int norm_weight_dtype,
                                                const float *__restrict__ freqs_cis,
                                                const void *__restrict__ positions,
                                                bool positions_i64,
                                                int64_t tokens,
                                                int64_t dim,
                                                float epsilon) {
    const int64_t token = blockIdx.x;
    const int lane = threadIdx.x;
    if (token >= tokens) {
        return;
    }

    extern __shared__ float smem[];
    const int64_t base = token * dim;
    float local_sum = 0.0f;
    for (int64_t i = lane; i < dim; i += blockDim.x) {
        const float v = load_scalar(input, base + i, input_dtype);
        local_sum += v * v;
    }
    smem[lane] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            smem[lane] += smem[lane + stride];
        }
        __syncthreads();
    }

    const float inv = rsqrtf(smem[0] / static_cast<float>(dim) + epsilon);
    const int64_t rope_start = dim - 64;
    for (int64_t i = lane; i < rope_start; i += blockDim.x) {
        const float w = load_scalar(norm_weight, i, norm_weight_dtype);
        const float v = load_scalar(input, base + i, input_dtype) * inv * w;
        store_scalar(input, base + i, input_dtype, v);
    }

    const int64_t pos = load_index(positions, token, positions_i64);
    for (int pair = lane; pair < 32; pair += blockDim.x) {
        const int64_t real_idx = rope_start + 2 * pair;
        const int64_t imag_idx = real_idx + 1;
        const float wr = load_scalar(norm_weight, real_idx, norm_weight_dtype);
        const float wi = load_scalar(norm_weight, imag_idx, norm_weight_dtype);
        const float xr = load_scalar(input, base + real_idx, input_dtype) * inv * wr;
        const float xi = load_scalar(input, base + imag_idx, input_dtype) * inv * wi;
        const float c = freqs_cis[pos * 64 + 2 * pair];
        const float s = freqs_cis[pos * 64 + 2 * pair + 1];
        store_scalar(input, base + real_idx, input_dtype, xr * c - xi * s);
        store_scalar(input, base + imag_idx, input_dtype, xr * s + xi * c);
    }
}

} // namespace

void launch_compress_fused_norm_rope(void *input,
                                      int input_dtype,
                                      const void *norm_weight,
                                      int norm_weight_dtype,
                                      const float *freqs_cis,
                                      const void *positions,
                                      bool positions_i64,
                                      int64_t tokens,
                                      int64_t dim,
                                      float epsilon,
                                      void *stream) {
    if (tokens <= 0 || dim <= 0) {
        return;
    }
    constexpr int threads = 256;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    compress_fused_norm_rope_kernel<<<static_cast<unsigned int>(tokens), threads, threads * sizeof(float), cuda_stream>>>(
        input, input_dtype, norm_weight, norm_weight_dtype, freqs_cis, positions, positions_i64, tokens, dim, epsilon);
}


} // namespace infinicore::op::deepseek_v4_compress_fused_norm_rope_kernel
