#include "deepseek_v4_fused_rope_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace infinicore::op::deepseek_v4_fused_rope_kernel_native {
namespace {

__device__ __forceinline__ float load_scalar(const void *__restrict__ ptr, int64_t idx, int dtype) {
    if (dtype == kDsv4BF16) {
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(ptr)[idx]);
    }
    if (dtype == kDsv4F16) {
        return __half2float(reinterpret_cast<const __half *>(ptr)[idx]);
    }
    return reinterpret_cast<const float *>(ptr)[idx];
}

__device__ __forceinline__ void store_scalar(void *__restrict__ ptr, int64_t idx, int dtype, float value) {
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

__global__ void fused_rope_kernel(void *__restrict__ tensor,
                                  int dtype,
                                  const float *__restrict__ freqs_cis,
                                  const void *__restrict__ positions,
                                  bool positions_i64,
                                  int64_t tokens,
                                  int64_t heads,
                                  int64_t stride_token,
                                  int64_t stride_head,
                                  bool inverse) {
    const int64_t row = blockIdx.x;
    const int pair = threadIdx.x;
    if (pair >= 32 || row >= tokens * heads) {
        return;
    }

    const int64_t token = row / heads;
    const int64_t head = row - token * heads;
    const int64_t base = token * stride_token + head * stride_head;
    const int64_t pos = load_index(positions, token, positions_i64);

    const int64_t real_idx = base + 2 * pair;
    const int64_t imag_idx = real_idx + 1;
    const float xr = load_scalar(tensor, real_idx, dtype);
    const float xi = load_scalar(tensor, imag_idx, dtype);
    const float c = freqs_cis[pos * 64 + 2 * pair];
    const float s = freqs_cis[pos * 64 + 2 * pair + 1];

    if (inverse) {
        store_scalar(tensor, real_idx, dtype, xr * c + xi * s);
        store_scalar(tensor, imag_idx, dtype, xi * c - xr * s);
    } else {
        store_scalar(tensor, real_idx, dtype, xr * c - xi * s);
        store_scalar(tensor, imag_idx, dtype, xr * s + xi * c);
    }
}

} // namespace

void launch_fused_rope(void *tensor,
                       int dtype,
                       const void *freqs_cis,
                       const void *positions,
                       bool positions_i64,
                       int64_t tokens,
                       int64_t heads,
                       int64_t stride_token,
                       int64_t stride_head,
                       bool inverse,
                       void *stream) {
    if (tokens <= 0 || heads <= 0) {
        return;
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    fused_rope_kernel<<<static_cast<unsigned int>(tokens * heads), 32, 0, cuda_stream>>>(
        tensor,
        dtype,
        reinterpret_cast<const float *>(freqs_cis),
        positions,
        positions_i64,
        tokens,
        heads,
        stride_token,
        stride_head,
        inverse);
}

} // namespace infinicore::op::deepseek_v4_fused_rope_kernel_native
