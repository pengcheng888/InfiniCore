#include "deepseek_v4_c4_compress_sglang_stateful_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

namespace infinicore::op::deepseek_v4_c4_compress_sglang_stateful_kernel_impl {
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

__global__ void c4_write_state_clear_non_boundary_kernel(void *__restrict__ output,
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
    const int64_t out_vecs = tokens * (dim / 4);
    const int64_t state_vecs = tokens * 4 * (dim / 4);
    if (idx < out_vecs) {
        const int64_t token = idx / (dim / 4);
        const int64_t d = (idx - token * (dim / 4)) * 4;
        const int64_t group = load_index(write_loc, token, write_loc_i64);
        const int64_t pos = load_index(positions, token, positions_i64);
        if (group < 0 || ((pos + 1) & 3) != 0) {
            store_vec4(output, token * dim + d, output_dtype, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
        }
    }
    if (idx >= state_vecs) {
        return;
    }

    const int64_t token = idx / (4 * (dim / 4));
    const int64_t rem = idx - token * 4 * (dim / 4);
    const int64_t channel = rem / (dim / 4);
    const int64_t d = (rem - channel * (dim / 4)) * 4;
    const int64_t group = load_index(write_loc, token, write_loc_i64);
    if (group < 0) {
        return;
    }
    const int64_t pos_mod = load_index(positions, token, positions_i64) & 3;
    const float4 value = load_vec4(kv_score, token * 4 * dim + channel * dim + d, kv_score_dtype);
    const int64_t state_idx = (((group * 4 + pos_mod) * 4 + channel) * dim + d);
    store_vec4(state, state_idx, state_dtype, value);
}

__global__ __launch_bounds__(256, 4) void c4_write_state_clear_non_boundary_dsv4_kernel(
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
    if (group < 0 || ((pos + 1) & 3) != 0) {
        for (int vec = threadIdx.x; vec < 128; vec += blockDim.x) {
            store_bf16_vec4(output, token * kDsv4HeadDim + static_cast<int64_t>(vec) * 4,
                            make_float4(0.0f, 0.0f, 0.0f, 0.0f));
        }
    }
    if (group < 0) {
        return;
    }

    const int pos_mod = pos & 3;
    for (int vec = threadIdx.x; vec < 512; vec += blockDim.x) {
        const int channel = vec >> 7;
        const int d_vec = vec & 127;
        const int64_t d = static_cast<int64_t>(d_vec) * 4;
        const float4 value = load_bf16_vec4(
            kv_score, token * 4 * kDsv4HeadDim + static_cast<int64_t>(channel) * kDsv4HeadDim + d);
        const int64_t state_idx =
            (((static_cast<int64_t>(group) * 4 + pos_mod) * 4 + channel) * kDsv4HeadDim + d);
        store_f32_vec4(state, state_idx, value);
    }
}

__global__ __launch_bounds__(128, 4) void c4_compress_boundary_sglang_kernel(void *__restrict__ output,
                                                                             int output_dtype,
                                                                             const void *__restrict__ state,
                                                                             int state_dtype,
                                                                             const void *__restrict__ ape,
                                                                             int ape_dtype,
                                                                             const void *__restrict__ write_loc,
                                                                             bool write_loc_i64,
                                                                             const void *__restrict__ extra_loc,
                                                                             bool extra_loc_i64,
                                                                             int64_t extra_cols,
                                                                             const void *__restrict__ positions,
                                                                             bool positions_i64,
                                                                             int64_t tokens,
                                                                             int64_t dim) {
    constexpr int kTileElements = 4;
    constexpr int kTileDim = kTileElements * kWarpThreads;
    const int64_t num_splits = dim / kTileDim;
    const int64_t global_warp = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpThreads;
    const int lane = threadIdx.x & (kWarpThreads - 1);
    const int64_t token = global_warp / num_splits;
    const int64_t split = global_warp - token * num_splits;
    if (token >= tokens) {
        return;
    }

    const int64_t group = load_index(write_loc, token, write_loc_i64);
    const int64_t pos = load_index(positions, token, positions_i64);
    if (group < 0 || ((pos + 1) & 3) != 0) {
        return;
    }

    const bool has_overlap = pos >= 7;
    int64_t prev_group = load_index(extra_loc, token * extra_cols, extra_loc_i64);
    if (prev_group < 0) {
        prev_group = 0;
    }
    const int64_t base_d = split * kTileDim + lane * kTileElements;

    float values[8][kTileElements];
    float scores[8][kTileElements];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float4 value_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 score_vec = make_float4(kNegInf, kNegInf, kNegInf, kNegInf);
        if (has_overlap) {
            value_vec = load_vec4(state, (((prev_group * 4 + i) * 4 + 0) * dim + base_d), state_dtype);
            score_vec = load_vec4(state, (((prev_group * 4 + i) * 4 + 2) * dim + base_d), state_dtype);
        }
        const float4 bias_vec = load_vec4(ape, static_cast<int64_t>(i) * dim + base_d, ape_dtype);
        values[i][0] = value_vec.x;
        values[i][1] = value_vec.y;
        values[i][2] = value_vec.z;
        values[i][3] = value_vec.w;
        scores[i][0] = score_vec.x + bias_vec.x;
        scores[i][1] = score_vec.y + bias_vec.y;
        scores[i][2] = score_vec.z + bias_vec.z;
        scores[i][3] = score_vec.w + bias_vec.w;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = i + 4;
        const float4 value_vec = load_vec4(state, (((group * 4 + i) * 4 + 1) * dim + base_d), state_dtype);
        const float4 score_vec = load_vec4(state, (((group * 4 + i) * 4 + 3) * dim + base_d), state_dtype);
        const float4 bias_vec = load_vec4(ape, static_cast<int64_t>(row) * dim + base_d, ape_dtype);
        values[row][0] = value_vec.x;
        values[row][1] = value_vec.y;
        values[row][2] = value_vec.z;
        values[row][3] = value_vec.w;
        scores[row][0] = score_vec.x + bias_vec.x;
        scores[row][1] = score_vec.y + bias_vec.y;
        scores[row][2] = score_vec.z + bias_vec.z;
        scores[row][3] = score_vec.w + bias_vec.w;
    }

    float4 result;
#pragma unroll
    for (int j = 0; j < kTileElements; ++j) {
        float max_score = scores[0][j];
#pragma unroll
        for (int i = 1; i < 8; ++i) {
            max_score = fmaxf(max_score, scores[i][j]);
        }
        float denom = 0.0f;
        float accum = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float p = fast_exp(scores[i][j] - max_score);
            denom += p;
            accum += values[i][j] * p;
        }
        if (j == 0) {
            result.x = accum / denom;
        } else if (j == 1) {
            result.y = accum / denom;
        } else if (j == 2) {
            result.z = accum / denom;
        } else {
            result.w = accum / denom;
        }
    }
    store_vec4(output, token * dim + base_d, output_dtype, result);
}

__global__ __launch_bounds__(128, 4) void c4_compress_boundary_dsv4_kernel(void *__restrict__ output,
                                                                           const void *__restrict__ state,
                                                                           const void *__restrict__ ape,
                                                                           const void *__restrict__ write_loc,
                                                                           const void *__restrict__ extra_loc,
                                                                           int64_t extra_cols,
                                                                           const void *__restrict__ positions,
                                                                           int64_t tokens) {
    const int64_t token = blockIdx.x;
    if (token >= tokens) {
        return;
    }

    __shared__ int group_s;
    __shared__ int pos_s;
    __shared__ int prev_group_s;
    if (threadIdx.x == 0) {
        group_s = reinterpret_cast<const int32_t *>(write_loc)[token];
        pos_s = reinterpret_cast<const int32_t *>(positions)[token];
        int prev_group = reinterpret_cast<const int32_t *>(extra_loc)[token * extra_cols];
        prev_group_s = prev_group < 0 ? 0 : prev_group;
    }
    __syncthreads();

    const int group = group_s;
    const int pos = pos_s;
    if (group < 0 || ((pos + 1) & 3) != 0) {
        return;
    }

    constexpr int kTileElements = 4;
    constexpr int kTileDim = kTileElements * kWarpThreads;
    const int split = threadIdx.x / kWarpThreads;
    const int lane = threadIdx.x & (kWarpThreads - 1);
    const int64_t base_d = static_cast<int64_t>(split) * kTileDim + lane * kTileElements;
    const bool has_overlap = pos >= 7;
    const int prev_group = prev_group_s;

    float values[8][kTileElements];
    float scores[8][kTileElements];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float4 value_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 score_vec = make_float4(kNegInf, kNegInf, kNegInf, kNegInf);
        if (has_overlap) {
            value_vec = load_f32_vec4(state, (((static_cast<int64_t>(prev_group) * 4 + i) * 4 + 0) * kDsv4HeadDim + base_d));
            score_vec = load_f32_vec4(state, (((static_cast<int64_t>(prev_group) * 4 + i) * 4 + 2) * kDsv4HeadDim + base_d));
        }
        const float4 bias_vec = load_bf16_vec4(ape, static_cast<int64_t>(i) * kDsv4HeadDim + base_d);
        values[i][0] = value_vec.x;
        values[i][1] = value_vec.y;
        values[i][2] = value_vec.z;
        values[i][3] = value_vec.w;
        scores[i][0] = score_vec.x + bias_vec.x;
        scores[i][1] = score_vec.y + bias_vec.y;
        scores[i][2] = score_vec.z + bias_vec.z;
        scores[i][3] = score_vec.w + bias_vec.w;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = i + 4;
        const float4 value_vec =
            load_f32_vec4(state, (((static_cast<int64_t>(group) * 4 + i) * 4 + 1) * kDsv4HeadDim + base_d));
        const float4 score_vec =
            load_f32_vec4(state, (((static_cast<int64_t>(group) * 4 + i) * 4 + 3) * kDsv4HeadDim + base_d));
        const float4 bias_vec = load_bf16_vec4(ape, static_cast<int64_t>(row) * kDsv4HeadDim + base_d);
        values[row][0] = value_vec.x;
        values[row][1] = value_vec.y;
        values[row][2] = value_vec.z;
        values[row][3] = value_vec.w;
        scores[row][0] = score_vec.x + bias_vec.x;
        scores[row][1] = score_vec.y + bias_vec.y;
        scores[row][2] = score_vec.z + bias_vec.z;
        scores[row][3] = score_vec.w + bias_vec.w;
    }

    float4 result;
#pragma unroll
    for (int j = 0; j < kTileElements; ++j) {
        float max_score = scores[0][j];
#pragma unroll
        for (int i = 1; i < 8; ++i) {
            max_score = fmaxf(max_score, scores[i][j]);
        }
        float denom = 0.0f;
        float accum = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float p = fast_exp(scores[i][j] - max_score);
            denom += p;
            accum += values[i][j] * p;
        }
        if (j == 0) {
            result.x = accum / denom;
        } else if (j == 1) {
            result.y = accum / denom;
        } else if (j == 2) {
            result.z = accum / denom;
        } else {
            result.w = accum / denom;
        }
    }
    store_bf16_vec4(output, token * kDsv4HeadDim + base_d, result);
}

__global__ __launch_bounds__(128, 4) void c4_single_token_sglang_kernel(void *__restrict__ output,
                                                                        int output_dtype,
                                                                        const void *__restrict__ kv_score,
                                                                        int kv_score_dtype,
                                                                        void *__restrict__ state,
                                                                        int state_dtype,
                                                                        const void *__restrict__ ape,
                                                                        int ape_dtype,
                                                                        const void *__restrict__ write_loc,
                                                                        bool write_loc_i64,
                                                                        const void *__restrict__ extra_loc,
                                                                        bool extra_loc_i64,
                                                                        int64_t extra_cols,
                                                                        const void *__restrict__ positions,
                                                                        bool positions_i64,
                                                                        int64_t dim) {
    constexpr int kTileElements = 4;
    constexpr int kTileDim = kTileElements * kWarpThreads;
    const int split = blockIdx.x;
    const int warp_id = threadIdx.x / kWarpThreads;
    const int lane = threadIdx.x & (kWarpThreads - 1);
    const int64_t base_d = static_cast<int64_t>(split) * kTileDim + lane * kTileElements;
    const int64_t group = load_index(write_loc, 0, write_loc_i64);
    const int64_t pos = load_index(positions, 0, positions_i64);

    if (group >= 0) {
        const int64_t pos_mod = pos & 3;
        const float4 value = load_vec4(kv_score, static_cast<int64_t>(warp_id) * dim + base_d, kv_score_dtype);
        const int64_t state_idx = (((group * 4 + pos_mod) * 4 + warp_id) * dim + base_d);
        store_vec4(state, state_idx, state_dtype, value);
    }
    __syncthreads();

    if (group < 0 || ((pos + 1) & 3) != 0) {
        if (warp_id == 0) {
            store_vec4(output, base_d, output_dtype, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
        }
        return;
    }
    if (warp_id != 0) {
        return;
    }

    const bool has_overlap = pos >= 7;
    int64_t prev_group = load_index(extra_loc, 0, extra_loc_i64);
    if (prev_group < 0) {
        prev_group = 0;
    }

    float values[8][kTileElements];
    float scores[8][kTileElements];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float4 value_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 score_vec = make_float4(kNegInf, kNegInf, kNegInf, kNegInf);
        if (has_overlap) {
            value_vec = load_vec4(state, (((prev_group * 4 + i) * 4 + 0) * dim + base_d), state_dtype);
            score_vec = load_vec4(state, (((prev_group * 4 + i) * 4 + 2) * dim + base_d), state_dtype);
        }
        const float4 bias_vec = load_vec4(ape, static_cast<int64_t>(i) * dim + base_d, ape_dtype);
        values[i][0] = value_vec.x;
        values[i][1] = value_vec.y;
        values[i][2] = value_vec.z;
        values[i][3] = value_vec.w;
        scores[i][0] = score_vec.x + bias_vec.x;
        scores[i][1] = score_vec.y + bias_vec.y;
        scores[i][2] = score_vec.z + bias_vec.z;
        scores[i][3] = score_vec.w + bias_vec.w;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = i + 4;
        const float4 value_vec = load_vec4(state, (((group * 4 + i) * 4 + 1) * dim + base_d), state_dtype);
        const float4 score_vec = load_vec4(state, (((group * 4 + i) * 4 + 3) * dim + base_d), state_dtype);
        const float4 bias_vec = load_vec4(ape, static_cast<int64_t>(row) * dim + base_d, ape_dtype);
        values[row][0] = value_vec.x;
        values[row][1] = value_vec.y;
        values[row][2] = value_vec.z;
        values[row][3] = value_vec.w;
        scores[row][0] = score_vec.x + bias_vec.x;
        scores[row][1] = score_vec.y + bias_vec.y;
        scores[row][2] = score_vec.z + bias_vec.z;
        scores[row][3] = score_vec.w + bias_vec.w;
    }

    float4 result;
#pragma unroll
    for (int j = 0; j < kTileElements; ++j) {
        float max_score = scores[0][j];
#pragma unroll
        for (int i = 1; i < 8; ++i) {
            max_score = fmaxf(max_score, scores[i][j]);
        }
        float denom = 0.0f;
        float accum = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float p = fast_exp(scores[i][j] - max_score);
            denom += p;
            accum += values[i][j] * p;
        }
        if (j == 0) {
            result.x = accum / denom;
        } else if (j == 1) {
            result.y = accum / denom;
        } else if (j == 2) {
            result.z = accum / denom;
        } else {
            result.w = accum / denom;
        }
    }
    store_vec4(output, base_d, output_dtype, result);
}

} // namespace

void launch_c4_compress_stateful_sglang(void *output,
                                        int output_dtype,
                                        const void *kv_score,
                                        int kv_score_dtype,
                                        void *compressor_state,
                                        int state_dtype,
                                        const void *ape,
                                        int ape_dtype,
                                        const void *write_loc,
                                        bool write_loc_i64,
                                        const void *extra_loc,
                                        bool extra_loc_i64,
                                        int64_t extra_cols,
                                        const void *positions,
                                        bool positions_i64,
                                        int64_t tokens,
                                        int64_t head_dim,
                                        void *stream) {
    if (tokens <= 0 || head_dim <= 0) {
        return;
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    constexpr int tile_dim = 4 * kWarpThreads;
    if (head_dim % tile_dim != 0) {
        return;
    }
    const bool use_dsv4_fast_path = head_dim == kDsv4HeadDim && output_dtype == kDsv4BF16 &&
                                    kv_score_dtype == kDsv4BF16 && state_dtype == kDsv4F32 &&
                                    ape_dtype == kDsv4BF16 && !write_loc_i64 && !extra_loc_i64 && !positions_i64;
    if (tokens == 1) {
        const int64_t num_splits = head_dim / tile_dim;
        c4_single_token_sglang_kernel<<<static_cast<unsigned int>(num_splits), 128, 0, cuda_stream>>>(
            output, output_dtype, kv_score, kv_score_dtype, compressor_state, state_dtype, ape, ape_dtype,
            write_loc, write_loc_i64, extra_loc, extra_loc_i64, extra_cols, positions, positions_i64, head_dim);
        return;
    }
    if (use_dsv4_fast_path) {
        c4_write_state_clear_non_boundary_dsv4_kernel<<<static_cast<unsigned int>(tokens), 256, 0, cuda_stream>>>(
            output, kv_score, compressor_state, write_loc, positions, tokens);
        c4_compress_boundary_dsv4_kernel<<<static_cast<unsigned int>(tokens), 128, 0, cuda_stream>>>(
            output, compressor_state, ape, write_loc, extra_loc, extra_cols, positions, tokens);
        return;
    }
    const int64_t state_numel = tokens * 4 * head_dim;
    const int64_t out_numel = tokens * head_dim;
    const int64_t write_numel = (state_numel > out_numel ? state_numel : out_numel) / 4;
    constexpr int threads = 256;
    const int64_t blocks = (write_numel + threads - 1) / threads;
    c4_write_state_clear_non_boundary_kernel<<<static_cast<unsigned int>(blocks), threads, 0, cuda_stream>>>(
        output, output_dtype, kv_score, kv_score_dtype, compressor_state, state_dtype, write_loc, write_loc_i64,
        positions, positions_i64, tokens, head_dim);

    constexpr int block_size = 128;
    const int64_t num_warps = tokens * (head_dim / tile_dim);
    const int64_t compress_blocks = (num_warps * kWarpThreads + block_size - 1) / block_size;
    c4_compress_boundary_sglang_kernel<<<static_cast<unsigned int>(compress_blocks), block_size, 0, cuda_stream>>>(
        output, output_dtype, compressor_state, state_dtype, ape, ape_dtype, write_loc, write_loc_i64, extra_loc,
        extra_loc_i64, extra_cols, positions, positions_i64, tokens, head_dim);
}

} // namespace infinicore::op::deepseek_v4_c4_compress_sglang_stateful_kernel_impl
