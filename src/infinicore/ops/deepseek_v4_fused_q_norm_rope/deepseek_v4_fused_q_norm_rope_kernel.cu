#include "deepseek_v4_fused_q_norm_rope_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace infinicore::op::deepseek_v4_fused_q_norm_rope_native {
namespace {

constexpr int64_t kHeadDim = 512;
constexpr int64_t kRopeDim = 64;
constexpr int kVecElems = 8;
constexpr int kWarpsPerBlock = 4;
#if defined(__HIP_PLATFORM_AMD__)
constexpr int kDeviceWarpSize = 64;
constexpr unsigned long long kFullWarpMask = 0xffffffffffffffffull;
#else
constexpr int kDeviceWarpSize = 32;
constexpr unsigned int kFullWarpMask = 0xffffffffu;
#endif
constexpr int kThreads = kWarpsPerBlock * kDeviceWarpSize;
constexpr int kVecsPerRow = kHeadDim / kVecElems;
constexpr int kVecsPerLane = kVecsPerRow / kDeviceWarpSize;
static_assert(kHeadDim % kVecElems == 0);
static_assert(kVecsPerRow % kDeviceWarpSize == 0);

struct alignas(16) Bf16x8 {
    __nv_bfloat16 values[kVecElems];
};

struct alignas(4) Bf16x2 {
    __nv_bfloat16 values[2];
};

__device__ __forceinline__ float load_bf16(const void *__restrict__ ptr, int64_t idx) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(ptr)[idx]);
}

__device__ __forceinline__ void store_bf16(void *__restrict__ ptr, int64_t idx, float value) {
    reinterpret_cast<__nv_bfloat16 *>(ptr)[idx] = __float2bfloat16(value);
    return;
}

__device__ __forceinline__ Bf16x8 load_bf16x8(const void *__restrict__ ptr, int64_t idx) {
    return *reinterpret_cast<const Bf16x8 *>(reinterpret_cast<const __nv_bfloat16 *>(ptr) + idx);
}

__device__ __forceinline__ Bf16x2 load_bf16x2(const void *__restrict__ ptr, int64_t idx) {
    return *reinterpret_cast<const Bf16x2 *>(reinterpret_cast<const __nv_bfloat16 *>(ptr) + idx);
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
    for (int offset = kDeviceWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullWarpMask, value, offset);
    }
    return __shfl_sync(kFullWarpMask, value, 0);
}

__device__ __forceinline__ bool row_info(int64_t rows,
                                         int64_t heads,
                                         int64_t q_input_stride_batch,
                                         int64_t q_out_stride_batch,
                                         int64_t *token,
                                         int64_t *input_base,
                                         int64_t *out_base) {
    const int warp_id = threadIdx.x / kDeviceWarpSize;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_id;
    if (row >= rows) {
        return false;
    }
    *token = row / heads;
    const int64_t head = row - *token * heads;
    *input_base = *token * q_input_stride_batch + head * kHeadDim;
    *out_base = *token * q_out_stride_batch + head * kHeadDim;
    return true;
}

__global__ void fused_q_norm_rope_vec_kernel(void *__restrict__ q_out,
                                             const void *__restrict__ q_input,
                                             const float *__restrict__ freqs_cis,
                                             const void *__restrict__ positions,
                                             bool positions_i64,
                                             int64_t rows,
                                             int64_t heads,
                                             int64_t q_input_stride_batch,
                                             int64_t q_out_stride_batch,
                                             float epsilon) {
    const int warp_id = threadIdx.x / kDeviceWarpSize;
    const int lane = threadIdx.x % kDeviceWarpSize;
    int64_t token = 0;
    int64_t input_base = 0;
    int64_t out_base = 0;
    if (!row_info(rows, heads, q_input_stride_batch, q_out_stride_batch, &token, &input_base, &out_base)) {
        return;
    }

    __shared__ Bf16x8 rope_smem[kWarpsPerBlock][kRopeDim / kVecElems];
    Bf16x8 input_vec[kVecsPerLane];
    float local_sum = 0.0f;

#pragma unroll
    for (int i = 0; i < kVecsPerLane; ++i) {
        const int64_t elem_offset = static_cast<int64_t>(i * kDeviceWarpSize + lane) * kVecElems;
        input_vec[i] = load_bf16x8(q_input, input_base + elem_offset);
#pragma unroll
        for (int j = 0; j < kVecElems; ++j) {
            const float v = __bfloat162float(input_vec[i].values[j]);
            local_sum += v * v;
        }
    }

    const float inv = rsqrtf(warp_sum(local_sum) / static_cast<float>(kHeadDim) + epsilon);
    const int64_t rope_start = kHeadDim - kRopeDim;

#pragma unroll
    for (int i = 0; i < kVecsPerLane; ++i) {
        const int64_t elem_offset = static_cast<int64_t>(i * kDeviceWarpSize + lane) * kVecElems;
        Bf16x8 norm_vec;
#pragma unroll
        for (int j = 0; j < kVecElems; ++j) {
            norm_vec.values[j] = __float2bfloat16(__bfloat162float(input_vec[i].values[j]) * inv);
        }
        if (elem_offset < rope_start) {
            store_bf16x8(q_out, out_base + elem_offset, norm_vec);
        } else {
            const int rope_offset = static_cast<int>(elem_offset - rope_start);
            rope_smem[warp_id][rope_offset / kVecElems] = norm_vec;
        }
    }
    __syncwarp();

    const int64_t pos = load_index(positions, token, positions_i64);
    for (int pair = lane; pair < kRopeDim / 2; pair += kDeviceWarpSize) {
        const int rope_idx = 2 * pair;
        const auto rope_values = load_bf16x2(rope_smem[warp_id], rope_idx);
        const float xr = __bfloat162float(rope_values.values[0]);
        const float xi = __bfloat162float(rope_values.values[1]);
        const float c = freqs_cis[pos * kRopeDim + 2 * pair];
        const float s = freqs_cis[pos * kRopeDim + 2 * pair + 1];
        store_bf16x2(q_out, out_base + rope_start + rope_idx, xr * c - xi * s, xr * s + xi * c);
    }
}

__global__ void fused_q_norm_rope_scalar_kernel(void *__restrict__ q_out,
                                                const void *__restrict__ q_input,
                                                const float *__restrict__ freqs_cis,
                                                const void *__restrict__ positions,
                                                bool positions_i64,
                                                int64_t rows,
                                                int64_t heads,
                                                int64_t q_input_stride_batch,
                                                int64_t q_out_stride_batch,
                                                float epsilon) {
    const int lane = threadIdx.x % kDeviceWarpSize;
    int64_t token = 0;
    int64_t input_base = 0;
    int64_t out_base = 0;
    if (!row_info(rows, heads, q_input_stride_batch, q_out_stride_batch, &token, &input_base, &out_base)) {
        return;
    }

    float local_sum = 0.0f;
    for (int64_t i = lane; i < kHeadDim; i += kDeviceWarpSize) {
        const float v = load_bf16(q_input, input_base + i);
        local_sum += v * v;
    }

    const float inv = rsqrtf(warp_sum(local_sum) / static_cast<float>(kHeadDim) + epsilon);
    const int64_t rope_start = kHeadDim - kRopeDim;
    for (int64_t i = lane; i < rope_start; i += kDeviceWarpSize) {
        store_bf16(q_out, out_base + i, load_bf16(q_input, input_base + i) * inv);
    }

    const int64_t pos = load_index(positions, token, positions_i64);
    for (int pair = lane; pair < kRopeDim / 2; pair += kDeviceWarpSize) {
        const int64_t real_idx = rope_start + 2 * pair;
        const int64_t imag_idx = real_idx + 1;
        const float xr = round_to_bf16_float(load_bf16(q_input, input_base + real_idx) * inv);
        const float xi = round_to_bf16_float(load_bf16(q_input, input_base + imag_idx) * inv);
        const float c = freqs_cis[pos * kRopeDim + 2 * pair];
        const float s = freqs_cis[pos * kRopeDim + 2 * pair + 1];
        store_bf16(q_out, out_base + real_idx, xr * c - xi * s);
        store_bf16(q_out, out_base + imag_idx, xr * s + xi * c);
    }
}

} // namespace

void launch_fused_q_norm_rope(void *q_out,
                              const void *q_input,
                              const float *freqs_cis,
                              const void *positions,
                              bool positions_i64,
                              int64_t tokens,
                              int64_t heads,
                              int64_t q_input_stride_batch,
                              int64_t q_out_stride_batch,
                              float epsilon,
                              void *stream) {
    if (tokens <= 0 || heads <= 0) {
        return;
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const int64_t rows = tokens * heads;
    const auto q_input_addr = reinterpret_cast<uintptr_t>(q_input);
    const auto q_out_addr = reinterpret_cast<uintptr_t>(q_out);
    const bool can_vectorize = q_input_addr % 16 == 0
        && q_out_addr % 16 == 0
        && q_input_stride_batch % kVecElems == 0
        && q_out_stride_batch % kVecElems == 0;
    const unsigned int blocks = static_cast<unsigned int>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    if (can_vectorize) {
        fused_q_norm_rope_vec_kernel<<<blocks, kThreads, 0, cuda_stream>>>(
            q_out, q_input, freqs_cis, positions, positions_i64, rows, heads, q_input_stride_batch, q_out_stride_batch, epsilon);
    } else {
        fused_q_norm_rope_scalar_kernel<<<blocks, kThreads, 0, cuda_stream>>>(
            q_out, q_input, freqs_cis, positions, positions_i64, rows, heads, q_input_stride_batch, q_out_stride_batch, epsilon);
    }
}

} // namespace infinicore::op::deepseek_v4_fused_q_norm_rope_native
