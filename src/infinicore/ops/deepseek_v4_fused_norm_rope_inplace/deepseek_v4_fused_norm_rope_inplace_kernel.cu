#include "deepseek_v4_fused_norm_rope_inplace_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace infinicore::op::deepseek_v4_fused_norm_rope_inplace_native {
namespace {

constexpr int64_t kHeadDim = 512;
constexpr int64_t kRopeDim = 64;
constexpr int kVecElems = 8;
constexpr int kWarpThreads = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kWarpKernelThreads = kWarpThreads * kWarpsPerBlock;
constexpr int kBlockKernelThreads = 256;
constexpr int kVecsPerRow = kHeadDim / kVecElems;
constexpr int kLocalSize = kVecsPerRow / kWarpThreads;
constexpr int kRopeSize = kRopeDim / kVecElems;
constexpr unsigned int kFullWarpMask = 0xffffffffu;
static_assert(kHeadDim % (kWarpThreads * kVecElems) == 0);
static_assert(kLocalSize * kVecElems * kWarpThreads == kHeadDim);
static_assert(kRopeDim == kWarpThreads * 2);
static_assert(kRopeSize <= kWarpThreads);

struct alignas(16) Bf16x8 {
    __nv_bfloat16 values[kVecElems];
};

struct alignas(4) Bf16x2 {
    __nv_bfloat16 values[2];
};

__device__ __forceinline__ float load_bf16(const void *__restrict__ ptr, int64_t idx) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(ptr)[idx]);
}

__device__ __forceinline__ Bf16x8 load_bf16x8(const void *__restrict__ ptr, int64_t idx) {
    return *reinterpret_cast<const Bf16x8 *>(reinterpret_cast<const __nv_bfloat16 *>(ptr) + idx);
}

__device__ __forceinline__ Bf16x2 load_bf16x2(const void *__restrict__ ptr, int64_t idx) {
    return *reinterpret_cast<const Bf16x2 *>(reinterpret_cast<const __nv_bfloat16 *>(ptr) + idx);
}

__device__ __forceinline__ void store_bf16(void *__restrict__ ptr, int64_t idx, float value) {
    reinterpret_cast<__nv_bfloat16 *>(ptr)[idx] = __float2bfloat16(value);
    return;
}

__device__ __forceinline__ void store_bf16x8(void *__restrict__ ptr, int64_t idx, const Bf16x8 &value) {
    *reinterpret_cast<Bf16x8 *>(reinterpret_cast<__nv_bfloat16 *>(ptr) + idx) = value;
    return;
}

__device__ __forceinline__ void store_bf16x2(void *__restrict__ ptr, int64_t idx, float real, float imag) {
    Bf16x2 value;
    value.values[0] = __float2bfloat16(real);
    value.values[1] = __float2bfloat16(imag);
    *reinterpret_cast<Bf16x2 *>(reinterpret_cast<__nv_bfloat16 *>(ptr) + idx) = value;
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

__global__ void fused_norm_rope_inplace_warp_kernel(void *__restrict__ input,
                                                    const void *__restrict__ norm_weight,
                                                    const float *__restrict__ freqs_cis,
                                                    const void *__restrict__ positions,
                                                    bool positions_i64,
                                                    int64_t tokens,
                                                    float epsilon) {
    const int warp_id = threadIdx.x / kWarpThreads;
    const int lane = threadIdx.x % kWarpThreads;
    const int64_t token = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_id;
    if (token >= tokens) {
        return;
    }

    __shared__ Bf16x8 rope_smem[kWarpsPerBlock][kRopeSize];
    Bf16x8 input_vec[kLocalSize];
    Bf16x8 weight_vec[kLocalSize];
    const int64_t base = token * kHeadDim;
    const int64_t rope_start = kHeadDim - kRopeDim;
    const int64_t pos = load_index(positions, token, positions_i64);
    float local_sum = 0.0f;

#pragma unroll
    for (int i = 0; i < kLocalSize; ++i) {
        const int64_t elem_offset = static_cast<int64_t>(i * kWarpThreads + lane) * kVecElems;
        input_vec[i] = load_bf16x8(input, base + elem_offset);
        weight_vec[i] = load_bf16x8(norm_weight, elem_offset);
#pragma unroll
        for (int j = 0; j < kVecElems; ++j) {
            const float v = __bfloat162float(input_vec[i].values[j]);
            local_sum += v * v;
        }
    }

    const float inv = rsqrtf(warp_sum(local_sum) / static_cast<float>(kHeadDim) + epsilon);

#pragma unroll
    for (int i = 0; i < kLocalSize; ++i) {
        const int64_t elem_offset = static_cast<int64_t>(i * kWarpThreads + lane) * kVecElems;
        Bf16x8 norm_vec;
#pragma unroll
        for (int j = 0; j < kVecElems; ++j) {
            const float x = __bfloat162float(input_vec[i].values[j]);
            const float w = __bfloat162float(weight_vec[i].values[j]);
            norm_vec.values[j] = __float2bfloat16(x * inv * w);
        }
        if (elem_offset < rope_start) {
            store_bf16x8(input, base + elem_offset, norm_vec);
        } else {
            const int rope_offset = static_cast<int>(elem_offset - rope_start);
            rope_smem[warp_id][rope_offset / kVecElems] = norm_vec;
        }
    }
    __syncwarp();

    const int rope_idx = 2 * lane;
    const auto rope_values = load_bf16x2(rope_smem[warp_id], rope_idx);
    const float xr = __bfloat162float(rope_values.values[0]);
    const float xi = __bfloat162float(rope_values.values[1]);
    const float c = freqs_cis[pos * kRopeDim + rope_idx];
    const float s = freqs_cis[pos * kRopeDim + rope_idx + 1];
    store_bf16x2(input, base + rope_start + rope_idx, xr * c - xi * s, xr * s + xi * c);
}

__global__ void fused_norm_rope_inplace_block_kernel(void *__restrict__ input,
                                                     const void *__restrict__ norm_weight,
                                                     const float *__restrict__ freqs_cis,
                                                     const void *__restrict__ positions,
                                                     bool positions_i64,
                                                     int64_t tokens,
                                                     float epsilon) {
    const int64_t token = blockIdx.x;
    const int lane = threadIdx.x;
    if (token >= tokens) {
        return;
    }

    extern __shared__ float smem[];
    const int64_t base = token * kHeadDim;
    float local_sum = 0.0f;
    for (int64_t i = lane; i < kHeadDim; i += blockDim.x) {
        const float v = load_bf16(input, base + i);
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

    const float inv = rsqrtf(smem[0] / static_cast<float>(kHeadDim) + epsilon);
    const int64_t rope_start = kHeadDim - kRopeDim;
    for (int64_t i = lane; i < rope_start; i += blockDim.x) {
        const float v = load_bf16(input, base + i) * inv * load_bf16(norm_weight, i);
        store_bf16(input, base + i, v);
    }

    const int64_t pos = load_index(positions, token, positions_i64);
    for (int pair = lane; pair < kRopeDim / 2; pair += blockDim.x) {
        const int64_t real_idx = rope_start + 2 * pair;
        const int64_t imag_idx = real_idx + 1;
        const float xr = round_to_bf16_float(load_bf16(input, base + real_idx) * inv * load_bf16(norm_weight, real_idx));
        const float xi = round_to_bf16_float(load_bf16(input, base + imag_idx) * inv * load_bf16(norm_weight, imag_idx));
        const float c = freqs_cis[pos * kRopeDim + 2 * pair];
        const float s = freqs_cis[pos * kRopeDim + 2 * pair + 1];
        store_bf16(input, base + real_idx, xr * c - xi * s);
        store_bf16(input, base + imag_idx, xr * s + xi * c);
    }
}

} // namespace

void launch_fused_norm_rope_inplace(void *input,
                                    const void *norm_weight,
                                    const float *freqs_cis,
                                    const void *positions,
                                    bool positions_i64,
                                    int64_t tokens,
                                    float epsilon,
                                    void *stream) {
    if (tokens <= 0) {
        return;
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const auto input_addr = reinterpret_cast<uintptr_t>(input);
    const auto weight_addr = reinterpret_cast<uintptr_t>(norm_weight);
    const bool can_vectorize = input_addr % 16 == 0 && weight_addr % 16 == 0;
    const unsigned int blocks = static_cast<unsigned int>((tokens + kWarpsPerBlock - 1) / kWarpsPerBlock);
    if (can_vectorize) {
        fused_norm_rope_inplace_warp_kernel<<<blocks, kWarpKernelThreads, 0, cuda_stream>>>(
            input, norm_weight, freqs_cis, positions, positions_i64, tokens, epsilon);
    } else {
        fused_norm_rope_inplace_block_kernel<<<static_cast<unsigned int>(tokens),
                                               kBlockKernelThreads,
                                               kBlockKernelThreads * sizeof(float),
                                               cuda_stream>>>(input, norm_weight, freqs_cis, positions, positions_i64, tokens, epsilon);
    }
}

} // namespace infinicore::op::deepseek_v4_fused_norm_rope_inplace_native
