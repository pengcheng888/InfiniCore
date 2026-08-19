#include "deepseek_v4_indexer_compress_norm_rope_store_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace infinicore::op::deepseek_v4_indexer_compress_norm_rope_store {
namespace {

constexpr int kHeadDim = 128;
constexpr int kRopeDim = 64;
constexpr int kVecSize = 4;
constexpr int kThreads = 256;
constexpr int kWarpThreads = 32;
constexpr int kNumWarps = kThreads / kWarpThreads;
constexpr int kRopeLaneBegin = 16;
constexpr int kValueBytesPerToken = 128;
constexpr int kScaleBytesPerToken = 4;
constexpr int kDsv4BF16 = 0;
constexpr float kFp8E4M3Max = 448.0f;
constexpr float kIndexerScaleFactor = 0.0022321429569274187f;
constexpr float kHadamardScale = 0.08838834764831845f;
constexpr unsigned int kFullWarpMask = 0xffffffffu;

struct Float4 {
    float v[4];
};

__device__ __forceinline__ float load_bf16_scalar(const void *__restrict__ ptr, int64_t idx) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(ptr)[idx]);
}

__device__ __forceinline__ float load_weight_scalar(const void *__restrict__ ptr, int64_t idx, int dtype) {
    if (dtype == kDsv4BF16) {
        return load_bf16_scalar(ptr, idx);
    }
    return reinterpret_cast<const float *>(ptr)[idx];
}

__device__ __forceinline__ int64_t load_index(const void *__restrict__ ptr, int64_t idx, bool i64) {
    return i64 ? reinterpret_cast<const int64_t *>(ptr)[idx]
               : static_cast<int64_t>(reinterpret_cast<const int32_t *>(ptr)[idx]);
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

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int offset = kWarpThreads / 2; offset > 0; offset >>= 1) {
#if defined(__HIP_PLATFORM_AMD__)
        value = fmaxf(value, __shfl_down(value, offset, kWarpThreads));
#else
        value = fmaxf(value, __shfl_down_sync(kFullWarpMask, value, offset, kWarpThreads));
#endif
    }
#if defined(__HIP_PLATFORM_AMD__)
    return __shfl(value, 0, kWarpThreads);
#else
    return __shfl_sync(kFullWarpMask, value, 0, kWarpThreads);
#endif
}

__device__ __forceinline__ float warp_xor(float value, int mask) {
#if defined(__HIP_PLATFORM_AMD__)
    return __shfl_xor(value, mask, kWarpThreads);
#else
    return __shfl_xor_sync(kFullWarpMask, value, mask, kWarpThreads);
#endif
}

__device__ __forceinline__ uint8_t fp8_e4m3_byte(float value) {
    value = fminf(fmaxf(value, -kFp8E4M3Max), kFp8E4M3Max);
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

template <int PageBits>
__global__ void indexer_compress_norm_rope_store_kernel(const void *__restrict__ kv,
                                                        const void *__restrict__ norm_weight,
                                                        int norm_weight_dtype,
                                                        const float *__restrict__ freqs_cis,
                                                        const void *__restrict__ positions,
                                                        bool positions_i64,
                                                        const void *__restrict__ out_loc,
                                                        bool out_loc_i64,
                                                        uint8_t *__restrict__ kvcache,
                                                        int64_t tokens,
                                                        int64_t kv_stride_batch,
                                                        int page_size,
                                                        int64_t page_bytes,
                                                        float epsilon) {
    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpThreads;
    const int lane = tid & (kWarpThreads - 1);
    const int64_t token = static_cast<int64_t>(blockIdx.x) * kNumWarps + warp_id;
    if (token >= tokens) {
        return;
    }

    const int64_t kv_base = token * kv_stride_batch;
    const int elem = lane * kVecSize;
    Float4 data;
    float local_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
        const float x = load_bf16_scalar(kv, kv_base + elem + i);
        data.v[i] = x;
        local_sum += x * x;
    }
    const float sum = warp_sum(local_sum);
    const float inv = rsqrtf(sum / static_cast<float>(kHeadDim) + epsilon);

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
        const float w = load_weight_scalar(norm_weight, elem + i, norm_weight_dtype);
        data.v[i] = data.v[i] * inv * w;
    }

    if (lane >= kRopeLaneBegin) {
        const int rope_base = (lane - kRopeLaneBegin) * kVecSize;
        const int64_t pos = load_index(positions, token, positions_i64);
        const float c0 = freqs_cis[pos * kRopeDim + rope_base + 0];
        const float s0 = freqs_cis[pos * kRopeDim + rope_base + 1];
        const float c1 = freqs_cis[pos * kRopeDim + rope_base + 2];
        const float s1 = freqs_cis[pos * kRopeDim + rope_base + 3];
        const float x0 = data.v[0];
        const float y0 = data.v[1];
        const float x1 = data.v[2];
        const float y1 = data.v[3];
        data.v[0] = x0 * c0 - y0 * s0;
        data.v[1] = x0 * s0 + y0 * c0;
        data.v[2] = x1 * c1 - y1 * s1;
        data.v[3] = x1 * s1 + y1 * c1;
    }

    {
        const float a0 = data.v[0];
        const float a1 = data.v[1];
        const float a2 = data.v[2];
        const float a3 = data.v[3];
        data.v[0] = a0 + a1;
        data.v[1] = a0 - a1;
        data.v[2] = a2 + a3;
        data.v[3] = a2 - a3;
    }
    {
        const float a0 = data.v[0];
        const float a1 = data.v[1];
        const float a2 = data.v[2];
        const float a3 = data.v[3];
        data.v[0] = a0 + a2;
        data.v[1] = a1 + a3;
        data.v[2] = a0 - a2;
        data.v[3] = a1 - a3;
    }
#pragma unroll
    for (int mask = 1; mask < kWarpThreads; mask <<= 1) {
#pragma unroll
        for (int i = 0; i < kVecSize; ++i) {
            const float other = warp_xor(data.v[i], mask);
            data.v[i] = (lane & mask) ? (other - data.v[i]) : (data.v[i] + other);
        }
    }

    float local_max = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
        data.v[i] *= kHadamardScale;
        local_max = fmaxf(local_max, fabsf(data.v[i]));
    }
    const float abs_max = warp_max(local_max);
    const float scale = fmaxf(abs_max, 1.0e-4f) * kIndexerScaleFactor;
    const float inv_scale = 1.0f / scale;

    const int64_t loc = load_index(out_loc, token, out_loc_i64);
    if (loc < 0) {
        return;
    }

    int64_t offset = 0;
    uint8_t *page_ptr = nullptr;
    if constexpr (PageBits >= 0) {
        constexpr int64_t kStaticPageSize = 1ll << PageBits;
        const int64_t page = loc >> PageBits;
        offset = loc & (kStaticPageSize - 1);
        page_ptr = kvcache + page * (kValueBytesPerToken + kScaleBytesPerToken) * kStaticPageSize;
    } else {
        const int64_t page = loc / page_size;
        offset = loc - page * static_cast<int64_t>(page_size);
        page_ptr = kvcache + page * page_bytes;
    }

    uint8_t *value_ptr = page_ptr + offset * kValueBytesPerToken;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
        value_ptr[elem + i] = fp8_e4m3_byte(data.v[i] * inv_scale);
    }
    if (lane < kScaleBytesPerToken) {
        union FloatBytes {
            float f;
            uint8_t b[4];
        } u;
        u.f = scale;
        uint8_t *scale_ptr = page_ptr + static_cast<int64_t>(kValueBytesPerToken) * page_size +
                             offset * kScaleBytesPerToken;
        scale_ptr[lane] = u.b[lane];
    }
}

} // namespace

void launch_indexer_compress_norm_rope_store(const void *kv,
                                             const void *norm_weight,
                                             int norm_weight_dtype,
                                             const float *freqs_cis,
                                             const void *positions,
                                             bool positions_i64,
                                             const void *out_loc,
                                             bool out_loc_i64,
                                             uint8_t *kvcache,
                                             int64_t tokens,
                                             int64_t kv_stride_batch,
                                             int page_size,
                                             int64_t page_bytes,
                                             float epsilon,
                                             void *stream) {
    if (tokens <= 0) {
        return;
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const auto blocks = static_cast<unsigned int>((tokens + kNumWarps - 1) / kNumWarps);
    if (page_size == 64) {
        indexer_compress_norm_rope_store_kernel<6><<<blocks, kThreads, 0, cuda_stream>>>(kv,
                                                                                        norm_weight,
                                                                                        norm_weight_dtype,
                                                                                        freqs_cis,
                                                                                        positions,
                                                                                        positions_i64,
                                                                                        out_loc,
                                                                                        out_loc_i64,
                                                                                        kvcache,
                                                                                        tokens,
                                                                                        kv_stride_batch,
                                                                                        page_size,
                                                                                        page_bytes,
                                                                                        epsilon);
        return;
    }
    indexer_compress_norm_rope_store_kernel<-1><<<blocks, kThreads, 0, cuda_stream>>>(kv,
                                                                                     norm_weight,
                                                                                     norm_weight_dtype,
                                                                                     freqs_cis,
                                                                                     positions,
                                                                                     positions_i64,
                                                                                     out_loc,
                                                                                     out_loc_i64,
                                                                                     kvcache,
                                                                                     tokens,
                                                                                     kv_stride_batch,
                                                                                     page_size,
                                                                                     page_bytes,
                                                                                     epsilon);
}

} // namespace infinicore::op::deepseek_v4_indexer_compress_norm_rope_store
