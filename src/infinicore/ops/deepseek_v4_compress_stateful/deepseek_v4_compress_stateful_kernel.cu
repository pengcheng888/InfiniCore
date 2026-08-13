#include "deepseek_v4_compress_stateful_kernel.hpp"

#include "../deepseek_v4_compress_common/deepseek_v4_compress_dtype.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace infinicore::op::deepseek_v4_compress_stateful_kernel {
namespace {

constexpr float kNegInf = -1.0e20f;

__device__ __forceinline__ float load_scalar(const void *__restrict__ ptr, int64_t idx, int dtype) {
    using namespace infinicore::op::deepseek_v4_compress_common;
    if (dtype == kDsv4BF16) {
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(ptr)[idx]);
    }
    if (dtype == kDsv4F16) {
        return __half2float(reinterpret_cast<const __half *>(ptr)[idx]);
    }
    return reinterpret_cast<const float *>(ptr)[idx];
}

__device__ __forceinline__ void store_scalar(void *__restrict__ ptr, int64_t idx, int dtype, float value) {
    using namespace infinicore::op::deepseek_v4_compress_common;
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

__device__ __forceinline__ float load_ape_c4(const void *__restrict__ ape,
                                             int ape_dtype,
                                             int ape_layout,
                                             int row,
                                             int64_t dim,
                                             int64_t d) {
    if (ape_layout == 0) {
        return load_scalar(ape, static_cast<int64_t>(row) * dim + d, ape_dtype);
    }
    if (row < 4) {
        return load_scalar(ape, static_cast<int64_t>(row) * 2 * dim + dim + d, ape_dtype);
    }
    return load_scalar(ape, static_cast<int64_t>(row - 4) * 2 * dim + d, ape_dtype);
}

__global__ void c4_write_state_zero_kernel(void *__restrict__ output,
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
    const int64_t out_numel = tokens * dim;
    const int64_t state_numel = tokens * 4 * dim;
    if (idx < out_numel) {
        store_scalar(output, idx, output_dtype, 0.0f);
    }
    if (idx >= state_numel) {
        return;
    }

    const int64_t token = idx / (4 * dim);
    const int64_t rem = idx - token * 4 * dim;
    const int64_t channel = rem / dim;
    const int64_t d = rem - channel * dim;
    const int64_t group = load_index(write_loc, token, write_loc_i64);
    if (group < 0) {
        return;
    }
    const int64_t pos_mod = load_index(positions, token, positions_i64) & 3;
    const float value = load_scalar(kv_score, token * 4 * dim + channel * dim + d, kv_score_dtype);
    const int64_t state_idx = (((group * 4 + pos_mod) * 4 + channel) * dim + d);
    store_scalar(state, state_idx, state_dtype, value);
}

__global__ void c4_compress_boundary_kernel(void *__restrict__ output,
                                            int output_dtype,
                                            const void *__restrict__ state,
                                            int state_dtype,
                                            const void *__restrict__ ape,
                                            int ape_dtype,
                                            int ape_layout,
                                            const void *__restrict__ write_loc,
                                            bool write_loc_i64,
                                            const void *__restrict__ extra_loc,
                                            bool extra_loc_i64,
                                            int64_t extra_cols,
                                            const void *__restrict__ positions,
                                            bool positions_i64,
                                            int64_t tokens,
                                            int64_t dim) {
    const int64_t token = blockIdx.x;
    const int64_t d = blockIdx.y;
    const int lane = threadIdx.x;
    if (token >= tokens || d >= dim || lane >= 8) {
        return;
    }
    const int64_t group = load_index(write_loc, token, write_loc_i64);
    const int64_t pos = load_index(positions, token, positions_i64);
    if (group < 0 || ((pos + 1) & 3) != 0) {
        return;
    }

    extern __shared__ float smem[];
    float *scores = smem;
    float *values = smem + 8;
    const bool has_overlap = pos >= 7;
    int64_t prev_group = load_index(extra_loc, token * extra_cols, extra_loc_i64);
    if (prev_group < 0) {
        prev_group = 0;
    }

    float score = kNegInf;
    float value = 0.0f;
    if (lane < 4) {
        if (has_overlap) {
            value = load_scalar(state, (((prev_group * 4 + lane) * 4 + 0) * dim + d), state_dtype);
            score = load_scalar(state, (((prev_group * 4 + lane) * 4 + 2) * dim + d), state_dtype) +
                    load_ape_c4(ape, ape_dtype, ape_layout, lane, dim, d);
        }
    } else {
        const int slot = lane - 4;
        value = load_scalar(state, (((group * 4 + slot) * 4 + 1) * dim + d), state_dtype);
        score = load_scalar(state, (((group * 4 + slot) * 4 + 3) * dim + d), state_dtype) +
                load_ape_c4(ape, ape_dtype, ape_layout, lane, dim, d);
    }
    scores[lane] = score;
    values[lane] = value;
    __syncthreads();

    float max_score = scores[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) {
        max_score = fmaxf(max_score, scores[i]);
    }
    float denom = 0.0f;
    float accum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float p = expf(scores[i] - max_score);
        denom += p;
        accum += values[i] * p;
    }
    if (lane == 0) {
        store_scalar(output, token * dim + d, output_dtype, accum / denom);
    }
}

__global__ void c128_write_state_zero_kernel(void *__restrict__ output,
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
    const int64_t out_numel = tokens * dim;
    const int64_t state_numel = tokens * 2 * dim;
    if (idx < out_numel) {
        store_scalar(output, idx, output_dtype, 0.0f);
    }
    if (idx >= state_numel) {
        return;
    }

    const int64_t token = idx / (2 * dim);
    const int64_t rem = idx - token * 2 * dim;
    const int64_t channel = rem / dim;
    const int64_t d = rem - channel * dim;
    const int64_t group = load_index(write_loc, token, write_loc_i64);
    if (group < 0) {
        return;
    }
    const int64_t pos_mod = load_index(positions, token, positions_i64) & 127;
    const float value = load_scalar(kv_score, token * 2 * dim + channel * dim + d, kv_score_dtype);
    const int64_t state_idx = (((group * 128 + pos_mod) * 2 + channel) * dim + d);
    store_scalar(state, state_idx, state_dtype, value);
}

__global__ void c128_compress_boundary_kernel(void *__restrict__ output,
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
    const int64_t token = blockIdx.x;
    const int64_t d = blockIdx.y;
    const int lane = threadIdx.x;
    if (token >= tokens || d >= dim || lane >= 128) {
        return;
    }
    const int64_t group = load_index(write_loc, token, write_loc_i64);
    const int64_t pos = load_index(positions, token, positions_i64);
    if (group < 0 || ((pos + 1) & 127) != 0) {
        return;
    }

    extern __shared__ float smem[];
    float *scores = smem;
    float *values = smem + 128;
    const float value = load_scalar(state, (((group * 128 + lane) * 2 + 0) * dim + d), state_dtype);
    const float score = load_scalar(state, (((group * 128 + lane) * 2 + 1) * dim + d), state_dtype) +
                        load_scalar(ape, static_cast<int64_t>(lane) * dim + d, ape_dtype);
    scores[lane] = score;
    values[lane] = value;
    __syncthreads();

    for (int stride = 64; stride > 0; stride >>= 1) {
        if (lane < stride) {
            scores[lane] = fmaxf(scores[lane], scores[lane + stride]);
        }
        __syncthreads();
    }
    const float max_score = scores[0];
    const float p = expf(score - max_score);
    scores[lane] = p;
    values[lane] = value * p;
    __syncthreads();

    for (int stride = 64; stride > 0; stride >>= 1) {
        if (lane < stride) {
            scores[lane] += scores[lane + stride];
            values[lane] += values[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        store_scalar(output, token * dim + d, output_dtype, values[0] / scores[0]);
    }
}

} // namespace

void launch_c4_compress_stateful(void *output,
                                  int output_dtype,
                                  const void *kv_score,
                                  int kv_score_dtype,
                                  void *compressor_state,
                                  int state_dtype,
                                  const void *ape,
                                  int ape_dtype,
                                  int ape_layout,
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
    const int64_t state_numel = tokens * 4 * head_dim;
    const int64_t out_numel = tokens * head_dim;
    const int64_t write_numel = state_numel > out_numel ? state_numel : out_numel;
    constexpr int threads = 256;
    const int64_t blocks = (write_numel + threads - 1) / threads;
    c4_write_state_zero_kernel<<<static_cast<unsigned int>(blocks), threads, 0, cuda_stream>>>(
        output, output_dtype, kv_score, kv_score_dtype, compressor_state, state_dtype, write_loc, write_loc_i64,
        positions, positions_i64, tokens, head_dim);
    dim3 grid(static_cast<unsigned int>(tokens), static_cast<unsigned int>(head_dim));
    c4_compress_boundary_kernel<<<grid, 8, 16 * sizeof(float), cuda_stream>>>(
        output, output_dtype, compressor_state, state_dtype, ape, ape_dtype, ape_layout, write_loc, write_loc_i64,
        extra_loc, extra_loc_i64, extra_cols, positions, positions_i64, tokens, head_dim);
}

void launch_c128_compress_stateful(void *output,
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
    const int64_t state_numel = tokens * 2 * head_dim;
    const int64_t out_numel = tokens * head_dim;
    const int64_t write_numel = state_numel > out_numel ? state_numel : out_numel;
    constexpr int threads = 256;
    const int64_t blocks = (write_numel + threads - 1) / threads;
    c128_write_state_zero_kernel<<<static_cast<unsigned int>(blocks), threads, 0, cuda_stream>>>(
        output, output_dtype, kv_score, kv_score_dtype, compressor_state, state_dtype, write_loc, write_loc_i64,
        positions, positions_i64, tokens, head_dim);
    dim3 grid(static_cast<unsigned int>(tokens), static_cast<unsigned int>(head_dim));
    c128_compress_boundary_kernel<<<grid, 128, 256 * sizeof(float), cuda_stream>>>(
        output, output_dtype, compressor_state, state_dtype, ape, ape_dtype, write_loc, write_loc_i64,
        positions, positions_i64, tokens, head_dim);
}


} // namespace infinicore::op::deepseek_v4_compress_stateful_kernel
