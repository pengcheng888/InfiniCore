#include "deepseek_v4_compress_norm_rope_store_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace infinicore::op::deepseek_v4_compress_norm_rope_store_native {
namespace {

constexpr int kHeadDim = 512;
constexpr int kNopeDim = 448;
constexpr int kRopeDim = 64;
constexpr int kThreads = 256;
constexpr int kWarpThreads = 32;
constexpr int kNumWarps = kThreads / kWarpThreads;
constexpr int kRopeWarp = 7;
constexpr int kValueBytesPerToken = 576;
constexpr int kScaleBytesPerToken = 8;
constexpr float kFp8E4M3Max = 448.0f;
constexpr unsigned int kFullWarpMask = 0xffffffffu;

struct Float2 {
    float x;
    float y;
};

__device__ __forceinline__ Float2 load_bf16_pair(const void *__restrict__ ptr, int64_t elem_idx) {
    const auto *bf16_ptr = reinterpret_cast<const __nv_bfloat16 *>(ptr) + elem_idx;
    const auto pair = *reinterpret_cast<const __nv_bfloat162 *>(bf16_ptr);
    return Float2{__low2float(pair), __high2float(pair)};
}

__device__ __forceinline__ void store_bf16_pair(void *__restrict__ ptr, int64_t elem_idx, float x, float y) {
    auto *bf16_ptr = reinterpret_cast<__nv_bfloat16 *>(ptr) + elem_idx;
    *reinterpret_cast<__nv_bfloat162 *>(bf16_ptr) = __floats2bfloat162_rn(x, y);
    return;
}

__device__ __forceinline__ float round_to_bf16_float(float value) {
    return __bfloat162float(__float2bfloat16(value));
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

__device__ __forceinline__ float warp_sum_8(float value) {
#pragma unroll
    for (int offset = kNumWarps / 2; offset > 0; offset >>= 1) {
#if defined(__HIP_PLATFORM_AMD__)
        value += __shfl_down(value, offset, kNumWarps);
#else
        value += __shfl_down_sync(kFullWarpMask, value, offset, kNumWarps);
#endif
    }
#if defined(__HIP_PLATFORM_AMD__)
    return __shfl(value, 0, kNumWarps);
#else
    return __shfl_sync(kFullWarpMask, value, 0, kNumWarps);
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

__device__ __forceinline__ uint8_t cast_to_ue8m0(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t exp = (bits >> 23) & 0xffu;
    const uint32_t mant = bits & 0x7fffffu;
    return static_cast<uint8_t>(exp + (mant != 0));
}

__device__ __forceinline__ float inv_scale_ue8m0(uint8_t exp) {
    return __uint_as_float(static_cast<uint32_t>(254 - static_cast<int>(exp)) << 23);
}

__device__ __forceinline__ uint8_t fp8_e4m3_byte(float value) {
    value = fminf(fmaxf(value, -kFp8E4M3Max), kFp8E4M3Max);
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ void store_fp8_pair(uint8_t *__restrict__ dst, int pair_idx, float x, float y) {
    const uint16_t packed = static_cast<uint16_t>(fp8_e4m3_byte(x)) |
                            (static_cast<uint16_t>(fp8_e4m3_byte(y)) << 8);
    reinterpret_cast<uint16_t *>(dst)[pair_idx] = packed;
    return;
}

template <int PageBits>
__global__ void compress_norm_rope_store_kernel(const void *__restrict__ kv,
                                                const void *__restrict__ norm_weight,
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
    const int64_t token = static_cast<int64_t>(blockIdx.x);
    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpThreads;
    const int lane = tid & (kWarpThreads - 1);
    if (token >= tokens) {
        return;
    }

    const int64_t kv_base = token * kv_stride_batch;
    const int elem = tid * 2;
    const Float2 input = load_bf16_pair(kv, kv_base + elem);
    float local_sum = input.x * input.x + input.y * input.y;
    local_sum = warp_sum(local_sum);

    __shared__ float partial_sums[kNumWarps];
    if (lane == 0) {
        partial_sums[warp_id] = local_sum;
    }
    __syncthreads();

    const float total_sum = warp_sum_8(partial_sums[lane % kNumWarps]);
    const float inv = rsqrtf(total_sum / static_cast<float>(kHeadDim) + epsilon);

    const Float2 weight = load_bf16_pair(norm_weight, elem);
    Float2 data{input.x * inv * weight.x, input.y * inv * weight.y};

    const int64_t loc = load_index(out_loc, token, out_loc_i64);
    if (loc < 0) {
        return;
    }

    int64_t offset = 0;
    uint8_t *page_ptr = nullptr;
    if constexpr (PageBits >= 0) {
        constexpr int64_t kStaticPageSize = 1ll << PageBits;
        constexpr int64_t kStaticBytes = (kValueBytesPerToken + kScaleBytesPerToken) * kStaticPageSize;
        constexpr int64_t kStaticPageBytes =
            ((kStaticBytes + kValueBytesPerToken - 1) / kValueBytesPerToken) * kValueBytesPerToken;
        const int64_t page = loc >> PageBits;
        offset = loc & (kStaticPageSize - 1);
        page_ptr = kvcache + page * kStaticPageBytes;
    } else {
        const int64_t page = loc / page_size;
        offset = loc - page * static_cast<int64_t>(page_size);
        page_ptr = kvcache + page * page_bytes;
    }
    uint8_t *value_ptr = page_ptr + offset * kValueBytesPerToken;

    if (warp_id == kRopeWarp) {
        const int rope_pair = lane;
        const int rope_idx = rope_pair * 2;
        const int64_t pos = load_index(positions, token, positions_i64);
        const float c = freqs_cis[pos * kRopeDim + rope_idx];
        const float s = freqs_cis[pos * kRopeDim + rope_idx + 1];
        const float real = data.x * c - data.y * s;
        const float imag = data.x * s + data.y * c;
        store_bf16_pair(value_ptr + kNopeDim, rope_idx, real, imag);
    } else {
        const float x = round_to_bf16_float(data.x);
        const float y = round_to_bf16_float(data.y);
        const float abs_max = warp_max(fmaxf(fabsf(x), fabsf(y)));
        const float scale_raw = fmaxf(1.0e-4f, abs_max) / kFp8E4M3Max;
        const uint8_t scale_ue8m0 = cast_to_ue8m0(scale_raw);
        const float inv_scale = inv_scale_ue8m0(scale_ue8m0);
        store_fp8_pair(value_ptr, tid, x * inv_scale, y * inv_scale);
        if (lane == 0) {
            uint8_t *scale_ptr = nullptr;
            if constexpr (PageBits >= 0) {
                scale_ptr = page_ptr + (static_cast<int64_t>(kValueBytesPerToken) << PageBits) +
                            offset * kScaleBytesPerToken;
            } else {
                scale_ptr = page_ptr + static_cast<int64_t>(kValueBytesPerToken) * page_size +
                            offset * kScaleBytesPerToken;
            }
            scale_ptr[warp_id] = scale_ue8m0;
        }
    }
}

} // namespace

void launch_compress_norm_rope_store(const void *kv,
                                     const void *norm_weight,
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
    if (page_size == 256) {
        compress_norm_rope_store_kernel<8><<<static_cast<unsigned int>(tokens), kThreads, 0, cuda_stream>>>(
            kv,
            norm_weight,
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
    compress_norm_rope_store_kernel<-1><<<static_cast<unsigned int>(tokens), kThreads, 0, cuda_stream>>>(
        kv,
        norm_weight,
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

} // namespace infinicore::op::deepseek_v4_compress_norm_rope_store_native
