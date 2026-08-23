#include "deepseek_v4_flashmla_cache_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace infinicore::op::deepseek_v4_flashmla_cache {
namespace {

constexpr int kIndexerDim = 128;
constexpr int kFlashNopeDim = 448;
constexpr int kFlashRopeDim = 64;
constexpr int kFlashInputDim = 512;
constexpr int kFlashValueBytesPerToken = 576;
constexpr int kFlashScaleBytesPerToken = 8;
constexpr float kFp8Max = 448.0f;

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
    return;
}

__device__ __forceinline__ int64_t load_index(const void *__restrict__ indices, int64_t idx, bool i64) {
    return i64 ? reinterpret_cast<const int64_t *>(indices)[idx]
               : static_cast<int64_t>(reinterpret_cast<const int32_t *>(indices)[idx]);
}

__device__ __forceinline__ float scale_from_max_abs(float max_abs) {
    return fmaxf(max_abs, 1.0e-4f) * 0.0022321429569274187f;
}

__device__ __forceinline__ uint8_t fp8_e4m3_byte(float value) {
    value = fminf(fmaxf(value, -kFp8Max), kFp8Max);
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ uint8_t flash_scale_exp_byte(float max_abs) {
    const float raw = scale_from_max_abs(max_abs);
    int exp_byte = static_cast<int>(ceilf(log2f(raw))) + 127;
    exp_byte = exp_byte < 0 ? 0 : (exp_byte > 255 ? 255 : exp_byte);
    return static_cast<uint8_t>(exp_byte);
}

__global__ void indexer_rotate_128_kernel(void *__restrict__ input,
                                          int dtype,
                                          int64_t rows,
                                          bool apply_scale) {
    const int64_t row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= rows || lane >= kIndexerDim) {
        return;
    }

    __shared__ float values[kIndexerDim];
    values[lane] = load_scalar(input, row * kIndexerDim + lane, dtype);
    __syncthreads();

#pragma unroll
    for (int span = 1; span < kIndexerDim; span <<= 1) {
        const int group = lane / (span << 1);
        const int offset = lane & ((span << 1) - 1);
        if (offset < span) {
            const int a_idx = group * (span << 1) + offset;
            const int b_idx = a_idx + span;
            const float a = values[a_idx];
            const float b = values[b_idx];
            values[a_idx] = a + b;
            values[b_idx] = a - b;
        }
        __syncthreads();
    }

    float out = values[lane];
    if (apply_scale) {
        out *= 0.08838834764831845f; // 1 / sqrt(128)
    }
    store_scalar(input, row * kIndexerDim + lane, dtype, out);
}

__global__ void store_indexer_raw_cache_kernel(const void *__restrict__ input,
                                               int input_dtype,
                                               uint8_t *__restrict__ cache,
                                               const void *__restrict__ indices,
                                               bool indices_i64,
                                               int64_t num_tokens,
                                               int page_size,
                                               int64_t page_bytes) {
    const int64_t token = blockIdx.x;
    const int lane = threadIdx.x;
    if (token >= num_tokens || lane >= kIndexerDim) {
        return;
    }

    const int64_t loc = load_index(indices, token, indices_i64);
    if (loc < 0) {
        return;
    }

    __shared__ float reduce[kIndexerDim];
    __shared__ float scale;
    const float value = load_scalar(input, token * kIndexerDim + lane, input_dtype);
    reduce[lane] = fabsf(value);
    __syncthreads();

#pragma unroll
    for (int stride = kIndexerDim / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reduce[lane] = fmaxf(reduce[lane], reduce[lane + stride]);
        }
        __syncthreads();
    }

    if (lane == 0) {
        scale = scale_from_max_abs(reduce[0]);
    }
    __syncthreads();

    const int64_t page = loc / page_size;
    const int64_t offset = loc - page * page_size;
    const int64_t value_base = page * page_bytes + offset * kIndexerDim;
    cache[value_base + lane] = fp8_e4m3_byte(value / scale);

    if (lane < 4) {
        union FloatBytes {
            float f;
            uint8_t b[4];
        } u;
        u.f = scale;
        const int64_t scale_base = page * page_bytes + static_cast<int64_t>(kIndexerDim) * page_size + offset * 4;
        cache[scale_base + lane] = u.b[lane];
    }
}

__global__ void store_flashmla_raw_cache_kernel(const void *__restrict__ input,
                                                int input_dtype,
                                                uint8_t *__restrict__ cache,
                                                const void *__restrict__ indices,
                                                bool indices_i64,
                                                int64_t num_tokens,
                                                int page_size,
                                                int64_t page_bytes) {
    const int64_t token = blockIdx.x;
    const int lane = threadIdx.x;
    if (token >= num_tokens || lane >= 512) {
        return;
    }

    const int64_t loc = load_index(indices, token, indices_i64);
    if (loc < 0) {
        return;
    }

    __shared__ float reduce[7][64];
    __shared__ float scales[7];
    __shared__ uint8_t scale_exp[7];

    for (int i = lane; i < kFlashNopeDim; i += blockDim.x) {
        const int group = i >> 6;
        const int sub = i & 63;
        const float value = load_scalar(input, token * kFlashInputDim + i, input_dtype);
        reduce[group][sub] = fabsf(value);
    }
    __syncthreads();

#pragma unroll
    for (int stride = 32; stride > 0; stride >>= 1) {
        for (int i = lane; i < kFlashNopeDim; i += blockDim.x) {
            const int group = i >> 6;
            const int sub = i & 63;
            if (sub < stride) {
                reduce[group][sub] = fmaxf(reduce[group][sub], reduce[group][sub + stride]);
            }
        }
        __syncthreads();
    }

    if (lane < 7) {
        const uint8_t exp_byte = flash_scale_exp_byte(reduce[lane][0]);
        scale_exp[lane] = exp_byte;
        scales[lane] = exp2f(static_cast<float>(static_cast<int>(exp_byte) - 127));
    }
    __syncthreads();

    const int64_t page = loc / page_size;
    const int64_t offset = loc - page * page_size;
    const int64_t token_base = page * page_bytes + offset * kFlashValueBytesPerToken;

    for (int i = lane; i < kFlashNopeDim; i += blockDim.x) {
        const int group = i >> 6;
        const float value = load_scalar(input, token * kFlashInputDim + i, input_dtype);
        cache[token_base + i] = fp8_e4m3_byte(value / scales[group]);
    }

    if (input_dtype != kDsv4F32) {
        for (int i = lane; i < kFlashRopeDim * 2; i += blockDim.x) {
            const uint8_t *input_bytes = reinterpret_cast<const uint8_t *>(input);
            const int64_t src = (token * kFlashInputDim + kFlashNopeDim) * 2 + i;
            cache[token_base + kFlashNopeDim + i] = input_bytes[src];
        }
    }

    if (lane < 7) {
        const int64_t scale_base = page * page_bytes + static_cast<int64_t>(kFlashValueBytesPerToken) * page_size +
                                   offset * kFlashScaleBytesPerToken;
        cache[scale_base + lane] = scale_exp[lane];
    }
}

} // namespace

void launch_indexer_rotate_128(void *input,
                               int dtype,
                               int64_t rows,
                               bool apply_scale,
                               void *stream) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    indexer_rotate_128_kernel<<<static_cast<unsigned int>(rows), kIndexerDim, 0, cuda_stream>>>(
        input, dtype, rows, apply_scale);
    return;
}

void launch_store_indexer_raw_cache(const void *input,
                                    int input_dtype,
                                    uint8_t *cache,
                                    const void *indices,
                                    bool indices_i64,
                                    int64_t num_tokens,
                                    int page_size,
                                    int64_t page_bytes,
                                    void *stream) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    store_indexer_raw_cache_kernel<<<static_cast<unsigned int>(num_tokens), kIndexerDim, 0, cuda_stream>>>(
        input, input_dtype, cache, indices, indices_i64, num_tokens, page_size, page_bytes);
    return;
}

void launch_store_flashmla_raw_cache(const void *input,
                                     int input_dtype,
                                     uint8_t *cache,
                                     const void *indices,
                                     bool indices_i64,
                                     int64_t num_tokens,
                                     int page_size,
                                     int64_t page_bytes,
                                     void *stream) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    store_flashmla_raw_cache_kernel<<<static_cast<unsigned int>(num_tokens), 256, 0, cuda_stream>>>(
        input, input_dtype, cache, indices, indices_i64, num_tokens, page_size, page_bytes);
    return;
}

} // namespace infinicore::op::deepseek_v4_flashmla_cache
