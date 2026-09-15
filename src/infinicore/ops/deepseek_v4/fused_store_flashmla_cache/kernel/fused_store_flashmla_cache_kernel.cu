#include "fused_store_flashmla_cache_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace infinicore::op::deepseek_v4::fused_store_flashmla_cache {
namespace {

constexpr int kNopeDim = 448;
constexpr int kRopeDim = 64;
constexpr int kInputDim = 512;
constexpr int kValueBytesPerToken = 576;
constexpr int kScaleBytesPerToken = 8;
constexpr float kFp8Max = 448.0f;

__device__ __forceinline__ float load_scalar(const void *__restrict__ ptr,
                                             int64_t idx,
                                             int dtype) {
    if (dtype == BF16) {
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(ptr)[idx]);
    }
    return __half2float(reinterpret_cast<const __half *>(ptr)[idx]);
}

__device__ __forceinline__ int64_t load_index(const void *__restrict__ indices,
                                              int64_t idx,
                                              bool i64) {
    return i64 ? reinterpret_cast<const int64_t *>(indices)[idx]
               : static_cast<int64_t>(reinterpret_cast<const int32_t *>(indices)[idx]);
}

__device__ __forceinline__ uint8_t fp8_e4m3_byte(float value) {
    value = fminf(fmaxf(value, -kFp8Max), kFp8Max);
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ uint8_t scale_exp_byte(float max_abs) {
    const float raw_scale = fmaxf(max_abs, 1.0e-4f) / kFp8Max;
    int exp_byte = static_cast<int>(ceilf(log2f(raw_scale))) + 127;
    exp_byte = exp_byte < 0 ? 0 : (exp_byte > 255 ? 255 : exp_byte);
    return static_cast<uint8_t>(exp_byte);
}

__global__ void fused_store_flashmla_cache_kernel(const void *__restrict__ input,
                                                  int input_dtype,
                                                  uint8_t *__restrict__ cache,
                                                  const void *__restrict__ indices,
                                                  bool indices_i64,
                                                  int64_t num_tokens,
                                                  int page_size,
                                                  int64_t page_bytes) {
    const int64_t token = blockIdx.x;
    const int lane = threadIdx.x;
    if (token >= num_tokens) {
        return;
    }

    const int64_t loc = load_index(indices, token, indices_i64);
    if (loc < 0) {
        return;
    }

    __shared__ float reduce[7][64];
    __shared__ float scales[7];
    __shared__ uint8_t scale_exp[7];

    for (int i = lane; i < kNopeDim; i += blockDim.x) {
        const int group = i >> 6;
        const int sub = i & 63;
        const float value = load_scalar(input, token * kInputDim + i, input_dtype);
        reduce[group][sub] = fabsf(value);
    }
    __syncthreads();

#pragma unroll
    for (int stride = 32; stride > 0; stride >>= 1) {
        for (int i = lane; i < kNopeDim; i += blockDim.x) {
            const int group = i >> 6;
            const int sub = i & 63;
            if (sub < stride) {
                reduce[group][sub] = fmaxf(reduce[group][sub], reduce[group][sub + stride]);
            }
        }
        __syncthreads();
    }

    if (lane < 7) {
        const uint8_t exp_byte = scale_exp_byte(reduce[lane][0]);
        scale_exp[lane] = exp_byte;
        scales[lane] = exp2f(static_cast<float>(static_cast<int>(exp_byte) - 127));
    }
    __syncthreads();

    const int64_t page = loc / page_size;
    const int64_t offset = loc - page * page_size;
    const int64_t token_base = page * page_bytes + offset * kValueBytesPerToken;

    for (int i = lane; i < kNopeDim; i += blockDim.x) {
        const int group = i >> 6;
        const float value = load_scalar(input, token * kInputDim + i, input_dtype);
        cache[token_base + i] = fp8_e4m3_byte(value / scales[group]);
    }

    const auto *input_bytes = reinterpret_cast<const uint8_t *>(input);
    for (int i = lane; i < kRopeDim * 2; i += blockDim.x) {
        const int64_t src = (token * kInputDim + kNopeDim) * 2 + i;
        cache[token_base + kNopeDim + i] = input_bytes[src];
    }

    if (lane < 7) {
        const int64_t scale_base = page * page_bytes
                                 + static_cast<int64_t>(kValueBytesPerToken) * page_size
                                 + offset * kScaleBytesPerToken;
        cache[scale_base + lane] = scale_exp[lane];
    }
}

} // namespace

void launch_fused_store_flashmla_cache(const void *input,
                                       int input_dtype,
                                       uint8_t *cache,
                                       const void *indices,
                                       bool indices_i64,
                                       int64_t num_tokens,
                                       int page_size,
                                       int64_t page_bytes,
                                       void *stream) {
    if (num_tokens <= 0) {
        return;
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    fused_store_flashmla_cache_kernel<<<static_cast<unsigned int>(num_tokens), 256, 0, cuda_stream>>>(
        input,
        input_dtype,
        cache,
        indices,
        indices_i64,
        num_tokens,
        page_size,
        page_bytes);
}

} // namespace infinicore::op::deepseek_v4::fused_store_flashmla_cache
