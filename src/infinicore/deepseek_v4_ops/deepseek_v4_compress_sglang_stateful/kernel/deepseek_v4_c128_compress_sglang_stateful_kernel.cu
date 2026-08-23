#include "deepseek_v4_c128_compress_sglang_stateful_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

namespace infinicore::op::deepseek_v4_c128_compress_sglang_stateful_kernel_impl {
namespace {

constexpr int kWarpThreads = 32;
constexpr int64_t kDsv4HeadDim = 512;
constexpr float kNegInf = -FLT_MAX;
constexpr int kDsv4BF16 = 0;
constexpr int kDsv4F16 = 1;
constexpr int kDsv4F32 = 2;

__device__ __forceinline__ float fast_exp(float value) {
    return __expf(value);
}

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

__device__ __forceinline__ float2 load_vec2(const void *__restrict__ ptr, int64_t idx, int dtype) {
    if (dtype == kDsv4BF16) {
        const auto value = reinterpret_cast<const __nv_bfloat162 *>(ptr)[idx / 2];
        return make_float2(__low2float(value), __high2float(value));
    }
    if (dtype == kDsv4F16) {
        const auto value = reinterpret_cast<const __half2 *>(ptr)[idx / 2];
        return make_float2(__low2float(value), __high2float(value));
    }
    return reinterpret_cast<const float2 *>(ptr)[idx / 2];
}

__device__ __forceinline__ void store_vec2(void *__restrict__ ptr, int64_t idx, int dtype, float2 value) {
    if (dtype == kDsv4BF16) {
        reinterpret_cast<__nv_bfloat162 *>(ptr)[idx / 2] = __floats2bfloat162_rn(value.x, value.y);
    } else if (dtype == kDsv4F16) {
        reinterpret_cast<__half2 *>(ptr)[idx / 2] = __floats2half2_rn(value.x, value.y);
    } else {
        reinterpret_cast<float2 *>(ptr)[idx / 2] = value;
    }
    return;
}

__device__ __forceinline__ float4 load_vec4(const void *__restrict__ ptr, int64_t idx, int dtype) {
    if (dtype == kDsv4BF16) {
        const auto *src = reinterpret_cast<const __nv_bfloat162 *>(ptr) + idx / 2;
        const auto lo = src[0];
        const auto hi = src[1];
        return make_float4(__low2float(lo), __high2float(lo), __low2float(hi), __high2float(hi));
    }
    if (dtype == kDsv4F16) {
        const auto *src = reinterpret_cast<const __half2 *>(ptr) + idx / 2;
        const auto lo = src[0];
        const auto hi = src[1];
        return make_float4(__low2float(lo), __high2float(lo), __low2float(hi), __high2float(hi));
    }
    return reinterpret_cast<const float4 *>(ptr)[idx / 4];
}

__device__ __forceinline__ void store_vec4(void *__restrict__ ptr, int64_t idx, int dtype, float4 value) {
    if (dtype == kDsv4BF16) {
        auto *dst = reinterpret_cast<__nv_bfloat162 *>(ptr) + idx / 2;
        dst[0] = __floats2bfloat162_rn(value.x, value.y);
        dst[1] = __floats2bfloat162_rn(value.z, value.w);
    } else if (dtype == kDsv4F16) {
        auto *dst = reinterpret_cast<__half2 *>(ptr) + idx / 2;
        dst[0] = __floats2half2_rn(value.x, value.y);
        dst[1] = __floats2half2_rn(value.z, value.w);
    } else {
        reinterpret_cast<float4 *>(ptr)[idx / 4] = value;
    }
    return;
}

__device__ __forceinline__ int64_t load_index(const void *__restrict__ ptr, int64_t idx, bool i64) {
    return i64 ? reinterpret_cast<const int64_t *>(ptr)[idx]
               : static_cast<int64_t>(reinterpret_cast<const int32_t *>(ptr)[idx]);
}

__device__ __forceinline__ float2 load_bf16_vec2(const void *__restrict__ ptr, int64_t idx) {
    const auto value = reinterpret_cast<const __nv_bfloat162 *>(ptr)[idx / 2];
    return make_float2(__low2float(value), __high2float(value));
}

__device__ __forceinline__ float4 load_bf16_vec4(const void *__restrict__ ptr, int64_t idx) {
    const auto *src = reinterpret_cast<const __nv_bfloat162 *>(ptr) + idx / 2;
    const auto lo = src[0];
    const auto hi = src[1];
    return make_float4(__low2float(lo), __high2float(lo), __low2float(hi), __high2float(hi));
}

__device__ __forceinline__ float2 load_f32_vec2(const void *__restrict__ ptr, int64_t idx) {
    return reinterpret_cast<const float2 *>(ptr)[idx / 2];
}

__device__ __forceinline__ float4 load_f32_vec4(const void *__restrict__ ptr, int64_t idx) {
    return reinterpret_cast<const float4 *>(ptr)[idx / 4];
}

__device__ __forceinline__ void store_bf16_vec2(void *__restrict__ ptr, int64_t idx, float2 value) {
    reinterpret_cast<__nv_bfloat162 *>(ptr)[idx / 2] = __floats2bfloat162_rn(value.x, value.y);
    return;
}

__device__ __forceinline__ void store_bf16_vec4(void *__restrict__ ptr, int64_t idx, float4 value) {
    auto *dst = reinterpret_cast<__nv_bfloat162 *>(ptr) + idx / 2;
    dst[0] = __floats2bfloat162_rn(value.x, value.y);
    dst[1] = __floats2bfloat162_rn(value.z, value.w);
    return;
}

__device__ __forceinline__ void store_f32_vec2(void *__restrict__ ptr, int64_t idx, float2 value) {
    reinterpret_cast<float2 *>(ptr)[idx / 2] = value;
    return;
}

__device__ __forceinline__ void store_f32_vec4(void *__restrict__ ptr, int64_t idx, float4 value) {
    reinterpret_cast<float4 *>(ptr)[idx / 4] = value;
    return;
}

__global__ void c128_write_state_clear_non_boundary_kernel(void *__restrict__ output,
                                                           int output_dtype,
                                                           const void *__restrict__ kv_score,
                                                           int kv_score_dtype,
                                                           void *__restrict__ state,
                                                           int state_dtype,
                                                           const void *__restrict__ write_loc,
                                                           bool write_loc_i64,
                                                           const void *__restrict__ positions,
                                                           bool positions_i64,
                                                           int64_t tokens,
                                                           int64_t dim) {
    const int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    const int64_t out_vecs = tokens * (dim / 2);
    const int64_t state_vecs = tokens * 2 * (dim / 2);
    if (idx < out_vecs) {
        const int64_t token = idx / (dim / 2);
        const int64_t d = (idx - token * (dim / 2)) * 2;
        const int64_t group = load_index(write_loc, token, write_loc_i64);
        const int64_t pos = load_index(positions, token, positions_i64);
        if (group < 0 || ((pos + 1) & 127) != 0) {
            store_vec2(output, token * dim + d, output_dtype, make_float2(0.0f, 0.0f));
        }
    }
    if (idx >= state_vecs) {
        return;
    }

    const int64_t token = idx / (2 * (dim / 2));
    const int64_t rem = idx - token * 2 * (dim / 2);
    const int64_t channel = rem / (dim / 2);
    const int64_t d = (rem - channel * (dim / 2)) * 2;
    const int64_t group = load_index(write_loc, token, write_loc_i64);
    if (group < 0) {
        return;
    }
    const int64_t pos_mod = load_index(positions, token, positions_i64) & 127;
    const float2 value = load_vec2(kv_score, token * 2 * dim + channel * dim + d, kv_score_dtype);
    const int64_t state_idx = (((group * 128 + pos_mod) * 2 + channel) * dim + d);
    store_vec2(state, state_idx, state_dtype, value);
}

__global__ __launch_bounds__(256, 4) void c128_write_state_clear_non_boundary_dsv4_kernel(
    void *__restrict__ output,
    const void *__restrict__ kv_score,
    void *__restrict__ state,
    const void *__restrict__ write_loc,
    const void *__restrict__ positions,
    int64_t tokens) {
    const int64_t token = blockIdx.x;
    if (token >= tokens) {
        return;
    }

    __shared__ int group_s;
    __shared__ int pos_s;
    if (threadIdx.x == 0) {
        group_s = reinterpret_cast<const int32_t *>(write_loc)[token];
        pos_s = reinterpret_cast<const int32_t *>(positions)[token];
    }
    __syncthreads();

    const int group = group_s;
    const int pos = pos_s;
    if (group < 0 || ((pos + 1) & 127) != 0) {
        for (int vec = threadIdx.x; vec < 256; vec += blockDim.x) {
            store_bf16_vec2(output, token * kDsv4HeadDim + static_cast<int64_t>(vec) * 2,
                            make_float2(0.0f, 0.0f));
        }
    }
    if (group < 0) {
        return;
    }

    const int pos_mod = pos & 127;
    for (int vec = threadIdx.x; vec < 512; vec += blockDim.x) {
        const int channel = vec >> 8;
        const int d_vec = vec & 255;
        const int64_t d = static_cast<int64_t>(d_vec) * 2;
        const float2 value = load_bf16_vec2(
            kv_score, token * 2 * kDsv4HeadDim + static_cast<int64_t>(channel) * kDsv4HeadDim + d);
        const int64_t state_idx =
            (((static_cast<int64_t>(group) * 128 + pos_mod) * 2 + channel) * kDsv4HeadDim + d);
        store_f32_vec2(state, state_idx, value);
    }
}

__device__ __forceinline__ float half_warp_reduce_max(float value, unsigned int mask) {
    const int lane = threadIdx.x & (kWarpThreads - 1);
    const int leader = lane < 16 ? 0 : 16;
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(mask, value, offset));
    }
    return __shfl_sync(mask, value, leader);
}

__device__ __forceinline__ float half_warp_reduce_sum(float value, unsigned int mask) {
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value;
}

__global__ __launch_bounds__(512, 2) void c128_compress_boundary_sglang_kernel(void *__restrict__ output,
                                                                               int output_dtype,
                                                                               const void *__restrict__ state,
                                                                               int state_dtype,
                                                                               const void *__restrict__ ape,
                                                                               int ape_dtype,
                                                                               const void *__restrict__ write_loc,
                                                                               bool write_loc_i64,
                                                                               const void *__restrict__ positions,
                                                                               bool positions_i64,
                                                                               int64_t tokens,
                                                                               int64_t dim) {
    constexpr int kTileElements = 2;
    constexpr int kElementsPerWarp = 8;
    constexpr int kNumWarps = 128 / kElementsPerWarp;
    constexpr int kTileDim = kTileElements * kWarpThreads;
    const int warp_id = threadIdx.x / kWarpThreads;
    const int lane = threadIdx.x & (kWarpThreads - 1);
    const int64_t num_splits = dim / kTileDim;
    const int64_t token = blockIdx.x / num_splits;
    const int64_t split = blockIdx.x - token * num_splits;
    if (token >= tokens) {
        return;
    }

    const int64_t group = load_index(write_loc, token, write_loc_i64);
    const int64_t pos = load_index(positions, token, positions_i64);
    if (group < 0 || ((pos + 1) & 127) != 0) {
        return;
    }

    __shared__ float s_max[kNumWarps][kWarpThreads + 1][kTileElements];
    __shared__ float s_sum[kNumWarps][kWarpThreads + 1][kTileElements];
    __shared__ float s_prod[kNumWarps][kWarpThreads + 1][kTileElements];

    const int64_t base_d = split * kTileDim + lane * kTileElements;
    const int row_offset = warp_id * kElementsPerWarp;
    float local_max[kTileElements];
    float local_sum[kTileElements];
    float local_prod[kTileElements];
#pragma unroll
    for (int j = 0; j < kTileElements; ++j) {
        local_max[j] = kNegInf;
        local_sum[j] = 0.0f;
        local_prod[j] = 0.0f;
    }

    float values[kTileElements][kElementsPerWarp];
    float scores[kTileElements][kElementsPerWarp];
#pragma unroll
    for (int i = 0; i < kElementsPerWarp; ++i) {
        const int row = row_offset + i;
        const float2 value_vec = load_vec2(state, (((group * 128 + row) * 2 + 0) * dim + base_d), state_dtype);
        const float2 score_vec = load_vec2(state, (((group * 128 + row) * 2 + 1) * dim + base_d), state_dtype);
        const float2 bias_vec = load_vec2(ape, static_cast<int64_t>(row) * dim + base_d, ape_dtype);
        values[0][i] = value_vec.x;
        values[1][i] = value_vec.y;
        scores[0][i] = score_vec.x + bias_vec.x;
        scores[1][i] = score_vec.y + bias_vec.y;
    }

#pragma unroll
    for (int j = 0; j < kTileElements; ++j) {
        float max_value = scores[j][0];
#pragma unroll
        for (int i = 1; i < kElementsPerWarp; ++i) {
            max_value = fmaxf(max_value, scores[j][i]);
        }
        float sum_exp = 0.0f;
        float sum_product = 0.0f;
#pragma unroll
        for (int i = 0; i < kElementsPerWarp; ++i) {
            const float p = fast_exp(scores[j][i] - max_value);
            sum_exp += p;
            sum_product += values[j][i] * p;
        }
        local_max[j] = max_value;
        local_sum[j] = sum_exp;
        local_prod[j] = sum_product;
    }

#pragma unroll
    for (int j = 0; j < kTileElements; ++j) {
        s_max[warp_id][lane][j] = local_max[j];
        s_sum[warp_id][lane][j] = local_sum[j];
        s_prod[warp_id][lane][j] = local_prod[j];
    }
    __syncthreads();

    constexpr int kReductionCount = kTileElements * kWarpThreads * kNumWarps;
    constexpr int kIteration = kReductionCount / 512;
    const unsigned int mask = (lane < 16) ? 0x0000ffffu : 0xffff0000u;
#pragma unroll
    for (int iter = 0; iter < kIteration; ++iter) {
        const int linear = iter * 512 + threadIdx.x;
        const int local_warp_id = linear % kNumWarps;
        const int local_elem_id = linear / kNumWarps;
        const int local_tile_id = local_elem_id % kTileElements;
        const int local_lane_id = local_elem_id / kTileElements;
        const float val_max = s_max[local_warp_id][local_lane_id][local_tile_id];
        const float exp_sum = s_sum[local_warp_id][local_lane_id][local_tile_id];
        const float product = s_prod[local_warp_id][local_lane_id][local_tile_id];
        const float global_max = half_warp_reduce_max(val_max, mask);
        const float rescale = fast_exp(val_max - global_max);
        const float global_exp_sum = half_warp_reduce_sum(exp_sum * rescale, mask);
        const float global_product = half_warp_reduce_sum(product * rescale, mask);
        if ((lane & 15) == 0) {
            store_scalar(output, token * dim + split * kTileDim + local_elem_id, output_dtype, global_product / global_exp_sum);
        }
    }
}

__global__ __launch_bounds__(512, 2) void c128_compress_boundary_dsv4_kernel(void *__restrict__ output,
                                                                             const void *__restrict__ state,
                                                                             const void *__restrict__ ape,
                                                                             const void *__restrict__ write_loc,
                                                                             const void *__restrict__ positions,
                                                                             int64_t tokens) {
    constexpr int kTileElements = 2;
    constexpr int kElementsPerWarp = 8;
    constexpr int kNumWarps = 128 / kElementsPerWarp;
    constexpr int kTileDim = kTileElements * kWarpThreads;
    constexpr int kNumSplits = kDsv4HeadDim / kTileDim;
    const int warp_id = threadIdx.x / kWarpThreads;
    const int lane = threadIdx.x & (kWarpThreads - 1);
    const int64_t token = blockIdx.x / kNumSplits;
    const int split = blockIdx.x - token * kNumSplits;
    if (token >= tokens) {
        return;
    }

    __shared__ int group_s;
    __shared__ int pos_s;
    if (threadIdx.x == 0) {
        group_s = reinterpret_cast<const int32_t *>(write_loc)[token];
        pos_s = reinterpret_cast<const int32_t *>(positions)[token];
    }
    __syncthreads();

    const int group = group_s;
    const int pos = pos_s;
    if (group < 0 || ((pos + 1) & 127) != 0) {
        return;
    }

    __shared__ float s_max[kNumWarps][kWarpThreads + 1][kTileElements];
    __shared__ float s_sum[kNumWarps][kWarpThreads + 1][kTileElements];
    __shared__ float s_prod[kNumWarps][kWarpThreads + 1][kTileElements];

    const int64_t base_d = static_cast<int64_t>(split) * kTileDim + lane * kTileElements;
    const int row_offset = warp_id * kElementsPerWarp;
    float local_max[kTileElements];
    float local_sum[kTileElements];
    float local_prod[kTileElements];
#pragma unroll
    for (int j = 0; j < kTileElements; ++j) {
        local_max[j] = kNegInf;
        local_sum[j] = 0.0f;
        local_prod[j] = 0.0f;
    }

    float values[kTileElements][kElementsPerWarp];
    float scores[kTileElements][kElementsPerWarp];
#pragma unroll
    for (int i = 0; i < kElementsPerWarp; ++i) {
        const int row = row_offset + i;
        const float2 value_vec =
            load_f32_vec2(state, (((static_cast<int64_t>(group) * 128 + row) * 2 + 0) * kDsv4HeadDim + base_d));
        const float2 score_vec =
            load_f32_vec2(state, (((static_cast<int64_t>(group) * 128 + row) * 2 + 1) * kDsv4HeadDim + base_d));
        const float2 bias_vec = load_bf16_vec2(ape, static_cast<int64_t>(row) * kDsv4HeadDim + base_d);
        values[0][i] = value_vec.x;
        values[1][i] = value_vec.y;
        scores[0][i] = score_vec.x + bias_vec.x;
        scores[1][i] = score_vec.y + bias_vec.y;
    }

#pragma unroll
    for (int j = 0; j < kTileElements; ++j) {
        float max_value = scores[j][0];
#pragma unroll
        for (int i = 1; i < kElementsPerWarp; ++i) {
            max_value = fmaxf(max_value, scores[j][i]);
        }
        float sum_exp = 0.0f;
        float sum_product = 0.0f;
#pragma unroll
        for (int i = 0; i < kElementsPerWarp; ++i) {
            const float p = fast_exp(scores[j][i] - max_value);
            sum_exp += p;
            sum_product += values[j][i] * p;
        }
        local_max[j] = max_value;
        local_sum[j] = sum_exp;
        local_prod[j] = sum_product;
    }

#pragma unroll
    for (int j = 0; j < kTileElements; ++j) {
        s_max[warp_id][lane][j] = local_max[j];
        s_sum[warp_id][lane][j] = local_sum[j];
        s_prod[warp_id][lane][j] = local_prod[j];
    }
    __syncthreads();

    constexpr int kReductionCount = kTileElements * kWarpThreads * kNumWarps;
    constexpr int kIteration = kReductionCount / 512;
    const unsigned int mask = (lane < 16) ? 0x0000ffffu : 0xffff0000u;
#pragma unroll
    for (int iter = 0; iter < kIteration; ++iter) {
        const int linear = iter * 512 + threadIdx.x;
        const int local_warp_id = linear % kNumWarps;
        const int local_elem_id = linear / kNumWarps;
        const int local_tile_id = local_elem_id % kTileElements;
        const int local_lane_id = local_elem_id / kTileElements;
        const float val_max = s_max[local_warp_id][local_lane_id][local_tile_id];
        const float exp_sum = s_sum[local_warp_id][local_lane_id][local_tile_id];
        const float product = s_prod[local_warp_id][local_lane_id][local_tile_id];
        const float global_max = half_warp_reduce_max(val_max, mask);
        const float rescale = fast_exp(val_max - global_max);
        const float global_exp_sum = half_warp_reduce_sum(exp_sum * rescale, mask);
        const float global_product = half_warp_reduce_sum(product * rescale, mask);
        if ((lane & 15) == 0) {
            store_scalar(output, token * kDsv4HeadDim + static_cast<int64_t>(split) * kTileDim + local_elem_id,
                         kDsv4BF16, global_product / global_exp_sum);
        }
    }
}

__global__ __launch_bounds__(512, 2) void c128_single_token_sglang_kernel(void *__restrict__ output,
                                                                          int output_dtype,
                                                                          const void *__restrict__ kv_score,
                                                                          int kv_score_dtype,
                                                                          void *__restrict__ state,
                                                                          int state_dtype,
                                                                          const void *__restrict__ ape,
                                                                          int ape_dtype,
                                                                          const void *__restrict__ write_loc,
                                                                          bool write_loc_i64,
                                                                          const void *__restrict__ positions,
                                                                          bool positions_i64,
                                                                          int64_t dim) {
    constexpr int kTileElements = 2;
    constexpr int kElementsPerWarp = 8;
    constexpr int kNumWarps = 128 / kElementsPerWarp;
    constexpr int kTileDim = kTileElements * kWarpThreads;
    const int warp_id = threadIdx.x / kWarpThreads;
    const int lane = threadIdx.x & (kWarpThreads - 1);
    const int split = blockIdx.x;
    const int64_t base_d = static_cast<int64_t>(split) * kTileDim + lane * kTileElements;
    const int64_t group = load_index(write_loc, 0, write_loc_i64);
    const int64_t pos = load_index(positions, 0, positions_i64);

    if (group >= 0 && warp_id < 2) {
        const int64_t pos_mod = pos & 127;
        const float2 value = load_vec2(kv_score, static_cast<int64_t>(warp_id) * dim + base_d, kv_score_dtype);
        const int64_t state_idx = (((group * 128 + pos_mod) * 2 + warp_id) * dim + base_d);
        store_vec2(state, state_idx, state_dtype, value);
    }
    __syncthreads();

    if (group < 0 || ((pos + 1) & 127) != 0) {
        if (warp_id == 0) {
            store_vec2(output, base_d, output_dtype, make_float2(0.0f, 0.0f));
        }
        return;
    }

    __shared__ float s_max[kNumWarps][kWarpThreads + 1][kTileElements];
    __shared__ float s_sum[kNumWarps][kWarpThreads + 1][kTileElements];
    __shared__ float s_prod[kNumWarps][kWarpThreads + 1][kTileElements];

    const int row_offset = warp_id * kElementsPerWarp;
    float local_max[kTileElements];
    float local_sum[kTileElements];
    float local_prod[kTileElements];
#pragma unroll
    for (int j = 0; j < kTileElements; ++j) {
        local_max[j] = kNegInf;
        local_sum[j] = 0.0f;
        local_prod[j] = 0.0f;
    }

    float values[kTileElements][kElementsPerWarp];
    float scores[kTileElements][kElementsPerWarp];
#pragma unroll
    for (int i = 0; i < kElementsPerWarp; ++i) {
        const int row = row_offset + i;
        const float2 value_vec = load_vec2(state, (((group * 128 + row) * 2 + 0) * dim + base_d), state_dtype);
        const float2 score_vec = load_vec2(state, (((group * 128 + row) * 2 + 1) * dim + base_d), state_dtype);
        const float2 bias_vec = load_vec2(ape, static_cast<int64_t>(row) * dim + base_d, ape_dtype);
        values[0][i] = value_vec.x;
        values[1][i] = value_vec.y;
        scores[0][i] = score_vec.x + bias_vec.x;
        scores[1][i] = score_vec.y + bias_vec.y;
    }

#pragma unroll
    for (int j = 0; j < kTileElements; ++j) {
        float max_value = scores[j][0];
#pragma unroll
        for (int i = 1; i < kElementsPerWarp; ++i) {
            max_value = fmaxf(max_value, scores[j][i]);
        }
        float sum_exp = 0.0f;
        float sum_product = 0.0f;
#pragma unroll
        for (int i = 0; i < kElementsPerWarp; ++i) {
            const float p = fast_exp(scores[j][i] - max_value);
            sum_exp += p;
            sum_product += values[j][i] * p;
        }
        local_max[j] = max_value;
        local_sum[j] = sum_exp;
        local_prod[j] = sum_product;
    }

#pragma unroll
    for (int j = 0; j < kTileElements; ++j) {
        s_max[warp_id][lane][j] = local_max[j];
        s_sum[warp_id][lane][j] = local_sum[j];
        s_prod[warp_id][lane][j] = local_prod[j];
    }
    __syncthreads();

    constexpr int kReductionCount = kTileElements * kWarpThreads * kNumWarps;
    constexpr int kIteration = kReductionCount / 512;
    const unsigned int mask = (lane < 16) ? 0x0000ffffu : 0xffff0000u;
#pragma unroll
    for (int iter = 0; iter < kIteration; ++iter) {
        const int linear = iter * 512 + threadIdx.x;
        const int local_warp_id = linear % kNumWarps;
        const int local_elem_id = linear / kNumWarps;
        const int local_tile_id = local_elem_id % kTileElements;
        const int local_lane_id = local_elem_id / kTileElements;
        const float val_max = s_max[local_warp_id][local_lane_id][local_tile_id];
        const float exp_sum = s_sum[local_warp_id][local_lane_id][local_tile_id];
        const float product = s_prod[local_warp_id][local_lane_id][local_tile_id];
        const float global_max = half_warp_reduce_max(val_max, mask);
        const float rescale = fast_exp(val_max - global_max);
        const float global_exp_sum = half_warp_reduce_sum(exp_sum * rescale, mask);
        const float global_product = half_warp_reduce_sum(product * rescale, mask);
        if ((lane & 15) == 0) {
            store_scalar(output, split * kTileDim + local_elem_id, output_dtype, global_product / global_exp_sum);
        }
    }
}

} // namespace

void launch_c128_compress_stateful_sglang(void *output,
                                          int output_dtype,
                                          const void *kv_score,
                                          int kv_score_dtype,
                                          void *compressor_state,
                                          int state_dtype,
                                          const void *ape,
                                          int ape_dtype,
                                          const void *write_loc,
                                          bool write_loc_i64,
                                          const void *positions,
                                          bool positions_i64,
                                          int64_t tokens,
                                          int64_t head_dim,
                                          void *stream) {
    if (tokens <= 0 || head_dim <= 0) {
        return;
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    constexpr int tile_dim = 2 * kWarpThreads;
    if (head_dim % tile_dim != 0) {
        return;
    }
    const bool use_dsv4_fast_path = head_dim == kDsv4HeadDim && output_dtype == kDsv4BF16 &&
                                    kv_score_dtype == kDsv4BF16 && state_dtype == kDsv4F32 &&
                                    ape_dtype == kDsv4BF16 && !write_loc_i64 && !positions_i64;
    if (tokens == 1) {
        const int64_t num_splits = head_dim / tile_dim;
        c128_single_token_sglang_kernel<<<static_cast<unsigned int>(num_splits), 512, 0, cuda_stream>>>(
            output, output_dtype, kv_score, kv_score_dtype, compressor_state, state_dtype, ape, ape_dtype,
            write_loc, write_loc_i64, positions, positions_i64, head_dim);
        return;
    }
    if (use_dsv4_fast_path) {
        c128_write_state_clear_non_boundary_dsv4_kernel<<<static_cast<unsigned int>(tokens), 256, 0, cuda_stream>>>(
            output, kv_score, compressor_state, write_loc, positions, tokens);
    } else {
        const int64_t state_numel = tokens * 2 * head_dim;
        const int64_t out_numel = tokens * head_dim;
        const int64_t write_numel = (state_numel > out_numel ? state_numel : out_numel) / 2;
        constexpr int threads = 256;
        const int64_t blocks = (write_numel + threads - 1) / threads;
        c128_write_state_clear_non_boundary_kernel<<<static_cast<unsigned int>(blocks), threads, 0, cuda_stream>>>(
            output, output_dtype, kv_score, kv_score_dtype, compressor_state, state_dtype, write_loc, write_loc_i64,
            positions, positions_i64, tokens, head_dim);
    }

    const int64_t compress_blocks = tokens * (head_dim / tile_dim);
    if (use_dsv4_fast_path) {
        c128_compress_boundary_dsv4_kernel<<<static_cast<unsigned int>(compress_blocks), 512, 0, cuda_stream>>>(
            output, compressor_state, ape, write_loc, positions, tokens);
    } else {
        c128_compress_boundary_sglang_kernel<<<static_cast<unsigned int>(compress_blocks), 512, 0, cuda_stream>>>(
            output, output_dtype, compressor_state, state_dtype, ape, ape_dtype, write_loc, write_loc_i64,
            positions, positions_i64, tokens, head_dim);
    }
}

} // namespace infinicore::op::deepseek_v4_c128_compress_sglang_stateful_kernel_impl
