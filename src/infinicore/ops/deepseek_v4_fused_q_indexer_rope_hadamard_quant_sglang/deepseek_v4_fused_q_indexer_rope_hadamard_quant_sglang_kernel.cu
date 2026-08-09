#include "deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace infinicore::op::deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang {
namespace {

constexpr int kHeadDim = 128;
constexpr int kRopeDim = 64;
constexpr int kVecSize = 4;
constexpr int kWarpThreads = 32;
constexpr int kBlockSize = 128;
constexpr int kNumWarps = kBlockSize / kWarpThreads;
constexpr uint32_t kFullWarpMask = 0xffffffffu;
constexpr float kFp8E4M3Max = 448.0f;

__device__ __forceinline__ float load_bf16(const void *__restrict__ ptr, int64_t idx) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(ptr)[idx]);
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

__global__ __launch_bounds__(kBlockSize, 16) void fused_q_indexer_rope_hadamard_quant_sglang_kernel(
    const void *__restrict__ q_input,
    uint8_t *__restrict__ q_fp8,
    const void *__restrict__ weight,
    float *__restrict__ weights_out,
    float weight_scale,
    const float *__restrict__ freqs_cis,
    const void *__restrict__ positions,
    bool positions_i64,
    int64_t rows,
    int64_t heads) {
    const int warp_id = threadIdx.x / kWarpThreads;
    const int lane_id = threadIdx.x % kWarpThreads;
    const int64_t work_id = static_cast<int64_t>(blockIdx.x) * kNumWarps + warp_id;
    if (work_id >= rows) {
        return;
    }

    const int64_t batch_id = work_id / heads;
    const int64_t position = positions_i64 ? reinterpret_cast<const int64_t *>(positions)[batch_id]
                                           : static_cast<int64_t>(reinterpret_cast<const int32_t *>(positions)[batch_id]);
    const float *freq = freqs_cis + static_cast<int64_t>(position) * kRopeDim;
    const bool is_rope_lane = lane_id >= kWarpThreads - (kRopeDim / kVecSize);

    float data[kVecSize];
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
        data[i] = load_bf16(q_input, work_id * kHeadDim + lane_id * kVecSize + i);
    }

    if (is_rope_lane) {
        const int rope_lane = lane_id - (kWarpThreads - (kRopeDim / kVecSize));
        const float fxr = freq[rope_lane * kVecSize + 0];
        const float fxi = freq[rope_lane * kVecSize + 1];
        const float fyr = freq[rope_lane * kVecSize + 2];
        const float fyi = freq[rope_lane * kVecSize + 3];
        const float x_r = data[0];
        const float x_i = data[1];
        const float y_r = data[2];
        const float y_i = data[3];
        data[0] = x_r * fxr - x_i * fxi;
        data[1] = x_r * fxi + x_i * fxr;
        data[2] = y_r * fyr - y_i * fyi;
        data[3] = y_r * fyi + y_i * fyr;
    }

    {
        const float a0 = data[0];
        const float a1 = data[1];
        const float a2 = data[2];
        const float a3 = data[3];
        data[0] = a0 + a1;
        data[1] = a0 - a1;
        data[2] = a2 + a3;
        data[3] = a2 - a3;
    }
    {
        const float a0 = data[0];
        const float a1 = data[1];
        const float a2 = data[2];
        const float a3 = data[3];
        data[0] = a0 + a2;
        data[1] = a1 + a3;
        data[2] = a0 - a2;
        data[3] = a1 - a3;
    }
#pragma unroll
    for (int mask = 1; mask < kWarpThreads; mask <<= 1) {
#pragma unroll
        for (int i = 0; i < kVecSize; ++i) {
            const float other = warp_xor(data[i], mask);
            data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
        }
    }

    constexpr float kHadamardScale = 0.08838834764831845f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
        data[i] *= kHadamardScale;
    }

    float local_max = fabsf(data[0]);
#pragma unroll
    for (int i = 1; i < kVecSize; ++i) {
        local_max = fmaxf(local_max, fabsf(data[i]));
    }
    const float abs_max = warp_max(local_max);
    const float scale = fmaxf(1.0e-4f, abs_max) / kFp8E4M3Max;
    const float inv_scale = 1.0f / scale;

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
        q_fp8[work_id * kHeadDim + lane_id * kVecSize + i] = fp8_e4m3_byte(data[i] * inv_scale);
    }
    weights_out[work_id] = load_bf16(weight, work_id) * weight_scale * scale;
}

} // namespace

void launch_fused_q_indexer_rope_hadamard_quant_sglang(const void *q_input,
                                                       uint8_t *q_fp8,
                                                       const void *weight,
                                                       float *weights_out,
                                                       float weight_scale,
                                                       const float *freqs_cis,
                                                       const void *positions,
                                                       bool positions_i64,
                                                       int64_t rows,
                                                       int64_t heads,
                                                       void *stream) {
    if (rows == 0) {
        return;
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const int64_t blocks = (rows + kNumWarps - 1) / kNumWarps;
    fused_q_indexer_rope_hadamard_quant_sglang_kernel<<<static_cast<unsigned int>(blocks), kBlockSize, 0, cuda_stream>>>(
        q_input,
        q_fp8,
        weight,
        weights_out,
        weight_scale,
        freqs_cis,
        positions,
        positions_i64,
        rows,
        heads);
    return;
}

} // namespace infinicore::op::deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang
