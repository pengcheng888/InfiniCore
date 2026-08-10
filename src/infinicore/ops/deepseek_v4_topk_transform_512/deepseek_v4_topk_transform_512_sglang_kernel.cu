/*
 * Adapted from SGLang's DeepSeek-V4 topk_transform_512 AOT kernel.
 * Original source:
 * sglang/kernels/aot/csrc/elementwise/deepseek_v4_topk.cu
 *
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#include "deepseek_v4_topk_transform_512_kernel.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace infinicore::op::deepseek_v4_topk_transform_512 {
namespace {

constexpr uint32_t kTopK = 512;
constexpr uint32_t kMaxTopK = 1024;
constexpr uint32_t kBlockSize = 512;
constexpr uint32_t kFastBlockSize = 256;
constexpr uint32_t kDsv4PageBits = 6;
constexpr uint32_t kDsv4PageSize = 1u << kDsv4PageBits;
constexpr uint32_t kDsv4OutputStride = kTopK;
constexpr size_t kSMEM = 48 * 1024;

static_assert(kSMEM % (2 * sizeof(int32_t)) == 0, "kSMEM must be a multiple of 8 bytes.");

struct TopKParams {
    const float *__restrict__ scores;
    const int32_t *__restrict__ seq_lens;
    const int32_t *__restrict__ page_table;
    int32_t *__restrict__ page_indices;
    int64_t score_stride;
    int64_t page_table_stride;
    int64_t output_stride;
    uint32_t page_bits;
};

struct TopKParamsDsv4 {
    const float *__restrict__ scores;
    const int32_t *__restrict__ seq_lens;
    const int32_t *__restrict__ page_table;
    int32_t *__restrict__ page_indices;
    int64_t score_stride;
    int64_t page_table_stride;
};

__device__ __forceinline__ uint8_t convert_to_uint8(float x) {
    __half h = __float2half_rn(x);
    uint16_t bits = __half_as_ushort(h);
    uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
    return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ uint32_t convert_to_uint32(float x) {
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ int32_t page_to_slot(const int32_t *__restrict__ page_table, uint32_t i, uint32_t page_bits) {
    const uint32_t mask = (1u << page_bits) - 1u;
    return (page_table[i >> page_bits] << page_bits) | static_cast<int32_t>(i & mask);
}

__device__ __forceinline__ int32_t page_to_slot_dsv4(const int32_t *__restrict__ page_table, uint32_t i) {
    return (page_table[i >> kDsv4PageBits] << kDsv4PageBits) | static_cast<int32_t>(i & (kDsv4PageSize - 1));
}

template <uint32_t BLOCK_SIZE>
__device__ void naive_paged_transform(int32_t length,
                                      uint32_t page_bits,
                                      const int32_t *__restrict__ page_table,
                                      int32_t *__restrict__ page_indices_out) {
    for (uint32_t i = threadIdx.x; i < kTopK; i += BLOCK_SIZE) {
        if (i < static_cast<uint32_t>(length)) {
            page_indices_out[i] = page_to_slot(page_table, i, page_bits);
        } else {
            page_indices_out[i] = -1;
        }
    }
}

template <uint32_t BLOCK_SIZE>
__device__ void naive_paged_transform_dsv4(int32_t length,
                                           const int32_t *__restrict__ page_table,
                                           int32_t *__restrict__ page_indices_out) {
    for (uint32_t i = threadIdx.x; i < kTopK; i += BLOCK_SIZE) {
        if (i < static_cast<uint32_t>(length)) {
            page_indices_out[i] = page_to_slot_dsv4(page_table, i);
        } else {
            page_indices_out[i] = -1;
        }
    }
}

__global__ __launch_bounds__(kFastBlockSize) void topk_transform_512_sglang_fast_kernel(const TopKParams params) {
    const auto row = blockIdx.x;
    const int32_t seq_len = params.seq_lens[row];
    const auto page_ptr = params.page_table + row * params.page_table_stride;
    auto indices_ptr = params.page_indices + row * params.output_stride;
    naive_paged_transform<kFastBlockSize>(seq_len, params.page_bits, page_ptr, indices_ptr);
}

__global__ __launch_bounds__(kFastBlockSize) void topk_transform_512_sglang_dsv4_fast_kernel(const TopKParamsDsv4 params) {
    const auto row = blockIdx.x;
    const int32_t seq_len = params.seq_lens[row];
    const auto page_ptr = params.page_table + row * params.page_table_stride;
    auto indices_ptr = params.page_indices + row * kDsv4OutputStride;
    naive_paged_transform_dsv4<kFastBlockSize>(seq_len, page_ptr, indices_ptr);
}

__device__ void radix_topk(const float *__restrict__ input, int32_t *__restrict__ output, uint32_t length) {
    constexpr uint32_t RADIX = 256;
    constexpr uint32_t BLOCK_SIZE = kBlockSize;
    constexpr uint32_t SMEM_INPUT_SIZE = kSMEM / (2 * sizeof(int32_t));

    alignas(128) __shared__ uint32_t _s_histogram_buf[2][RADIX + 32];
    alignas(128) __shared__ uint32_t s_counter;
    alignas(128) __shared__ uint32_t s_threshold_bin_id;
    alignas(128) __shared__ uint32_t s_num_input[2];
    alignas(128) __shared__ int32_t s_last_remain;

    extern __shared__ uint32_t s_input_idx[][SMEM_INPUT_SIZE];

    const uint32_t tx = threadIdx.x;
    uint32_t remain_topk = kTopK;
    auto &s_histogram = _s_histogram_buf[0];

    const auto run_cumsum = [&] {
#pragma unroll 8
        for (int32_t i = 0; i < 8; ++i) {
            if (tx < RADIX) {
                const auto j = 1 << i;
                const auto k = i & 1;
                auto value = _s_histogram_buf[k][tx];
                if (tx + j < RADIX) {
                    value += _s_histogram_buf[k][tx + j];
                }
                _s_histogram_buf[k ^ 1][tx] = value;
            }
            __syncthreads();
        }
    };

    if (tx < RADIX + 1) {
        s_histogram[tx] = 0;
    }
    __syncthreads();
    for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
        const auto bin = convert_to_uint8(input[idx]);
        atomicAdd(&s_histogram[bin], 1);
    }
    __syncthreads();
    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
        s_threshold_bin_id = tx;
        s_num_input[0] = 0;
        s_counter = 0;
    }
    __syncthreads();

    {
        const auto threshold_bin = s_threshold_bin_id;
        remain_topk -= s_histogram[threshold_bin + 1];
        if (remain_topk == 0) {
            for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
                const uint32_t bin = convert_to_uint8(input[idx]);
                if (bin > threshold_bin) {
                    const auto pos = atomicAdd(&s_counter, 1);
                    output[pos] = static_cast<int32_t>(idx);
                }
            }
            __syncthreads();
            return;
        }
        __syncthreads();
        if (tx < RADIX + 1) {
            s_histogram[tx] = 0;
        }
        __syncthreads();

        for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
            const float raw_input = input[idx];
            const uint32_t bin = convert_to_uint8(raw_input);
            if (bin > threshold_bin) {
                const auto pos = atomicAdd(&s_counter, 1);
                output[pos] = static_cast<int32_t>(idx);
            } else if (bin == threshold_bin) {
                const auto pos = atomicAdd(&s_num_input[0], 1);
                if (pos < SMEM_INPUT_SIZE) {
                    s_input_idx[0][pos] = idx;
                    const auto bin32 = convert_to_uint32(raw_input);
                    const auto sub_bin = (bin32 >> 24) & 0xFF;
                    atomicAdd(&s_histogram[sub_bin], 1);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll 4
    for (int round = 0; round < 4; ++round) {
        const auto r_idx = round % 2;
        const auto raw_num_input = s_num_input[r_idx];
        const auto num_input = raw_num_input < SMEM_INPUT_SIZE ? raw_num_input : SMEM_INPUT_SIZE;

        run_cumsum();
        if (tx < RADIX && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
            s_threshold_bin_id = tx;
            s_num_input[r_idx ^ 1] = 0;
            s_last_remain = static_cast<int32_t>(remain_topk - s_histogram[tx + 1]);
        }
        __syncthreads();

        const auto threshold_bin = s_threshold_bin_id;
        remain_topk -= s_histogram[threshold_bin + 1];

        if (remain_topk == 0) {
            for (uint32_t i = tx; i < num_input; i += BLOCK_SIZE) {
                const auto idx = s_input_idx[r_idx][i];
                const auto offset = 24 - round * 8;
                const auto bin = (convert_to_uint32(input[idx]) >> offset) & 0xFF;
                if (bin > threshold_bin) {
                    const auto pos = atomicAdd(&s_counter, 1);
                    output[pos] = static_cast<int32_t>(idx);
                }
            }
            __syncthreads();
            break;
        }
        __syncthreads();
        if (tx < RADIX + 1) {
            s_histogram[tx] = 0;
        }
        __syncthreads();
        for (uint32_t i = tx; i < num_input; i += BLOCK_SIZE) {
            const auto idx = s_input_idx[r_idx][i];
            const auto raw_input = input[idx];
            const auto offset = 24 - round * 8;
            const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
            if (bin > threshold_bin) {
                const auto pos = atomicAdd(&s_counter, 1);
                output[pos] = static_cast<int32_t>(idx);
            } else if (bin == threshold_bin) {
                if (round == 3) {
                    const auto pos = atomicAdd(&s_last_remain, -1);
                    if (pos > 0) {
                        output[kTopK - pos] = static_cast<int32_t>(idx);
                    }
                } else {
                    const auto pos = atomicAdd(&s_num_input[r_idx ^ 1], 1);
                    if (pos < SMEM_INPUT_SIZE) {
                        s_input_idx[r_idx ^ 1][pos] = idx;
                        const auto bin32 = convert_to_uint32(raw_input);
                        const auto sub_bin = (bin32 >> (offset - 8)) & 0xFF;
                        atomicAdd(&s_histogram[sub_bin], 1);
                    }
                }
            }
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(kBlockSize) void topk_transform_512_sglang_kernel(const TopKParams params) {
    const auto row = blockIdx.x;
    const int32_t seq_len = params.seq_lens[row];

    const auto score_ptr = params.scores + row * params.score_stride;
    const auto page_ptr = params.page_table + row * params.page_table_stride;
    auto indices_ptr = params.page_indices + row * params.output_stride;

    if (seq_len <= static_cast<int32_t>(kTopK)) {
        naive_paged_transform<kBlockSize>(seq_len, params.page_bits, page_ptr, indices_ptr);
        return;
    }

    __shared__ int32_t s_topk_indices[kMaxTopK];
    radix_topk(score_ptr, s_topk_indices, static_cast<uint32_t>(seq_len));

    __syncthreads();
    for (uint32_t i = threadIdx.x; i < kTopK; i += kBlockSize) {
        const auto raw = s_topk_indices[i];
        indices_ptr[i] = page_to_slot(page_ptr, static_cast<uint32_t>(raw), params.page_bits);
    }
}

__global__ __launch_bounds__(kBlockSize) void topk_transform_512_sglang_dsv4_kernel(const TopKParamsDsv4 params) {
    const auto row = blockIdx.x;
    const int32_t seq_len = params.seq_lens[row];

    const auto score_ptr = params.scores + row * params.score_stride;
    const auto page_ptr = params.page_table + row * params.page_table_stride;
    auto indices_ptr = params.page_indices + row * kDsv4OutputStride;

    if (seq_len <= static_cast<int32_t>(kTopK)) {
        naive_paged_transform_dsv4<kBlockSize>(seq_len, page_ptr, indices_ptr);
        return;
    }

    __shared__ int32_t s_topk_indices[kMaxTopK];
    radix_topk(score_ptr, s_topk_indices, static_cast<uint32_t>(seq_len));

    __syncthreads();
    for (uint32_t i = threadIdx.x; i < kTopK; i += kBlockSize) {
        const auto raw = s_topk_indices[i];
        indices_ptr[i] = page_to_slot_dsv4(page_ptr, static_cast<uint32_t>(raw));
    }
}

void setup_kernel_smem_once() {
    static const auto result = [] {
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
        return cudaFuncSetAttribute(
            reinterpret_cast<const void *>(topk_transform_512_sglang_kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kSMEM);
#else
        return cudaFuncSetAttribute(
            topk_transform_512_sglang_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kSMEM);
#endif
    }();
    (void)result;
}

void setup_dsv4_kernel_smem_once() {
    static const auto result = [] {
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
        return cudaFuncSetAttribute(
            reinterpret_cast<const void *>(topk_transform_512_sglang_dsv4_kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kSMEM);
#else
        return cudaFuncSetAttribute(
            topk_transform_512_sglang_dsv4_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kSMEM);
#endif
    }();
    (void)result;
}

uint32_t page_bits_from_size(int page_size) {
    return static_cast<uint32_t>(__builtin_ctz(static_cast<unsigned int>(page_size)));
}

} // namespace

void launch_topk_transform_512_sglang(const float *scores,
                                      int64_t score_stride0,
                                      const int32_t *seq_lens,
                                      const int32_t *page_table,
                                      int64_t page_table_stride0,
                                      int32_t *out_page_indices,
                                      int64_t out_stride0,
                                      int64_t batch,
                                      int64_t max_seq_len,
                                      int page_size,
                                      void *stream) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (page_size == static_cast<int>(kDsv4PageSize) && out_stride0 == static_cast<int64_t>(kDsv4OutputStride)) {
        const TopKParamsDsv4 params{
            scores,
            seq_lens,
            page_table,
            out_page_indices,
            score_stride0,
            page_table_stride0,
        };
        if (max_seq_len <= static_cast<int64_t>(kTopK)) {
            topk_transform_512_sglang_dsv4_fast_kernel<<<static_cast<unsigned int>(batch), kFastBlockSize, 0, cuda_stream>>>(params);
            return;
        }

        setup_dsv4_kernel_smem_once();
        topk_transform_512_sglang_dsv4_kernel<<<static_cast<unsigned int>(batch), kBlockSize, kSMEM, cuda_stream>>>(params);
        return;
    }

    const TopKParams params{
        scores,
        seq_lens,
        page_table,
        out_page_indices,
        score_stride0,
        page_table_stride0,
        out_stride0,
        page_bits_from_size(page_size),
    };
    if (max_seq_len <= static_cast<int64_t>(kTopK)) {
        topk_transform_512_sglang_fast_kernel<<<static_cast<unsigned int>(batch), kFastBlockSize, 0, cuda_stream>>>(params);
        return;
    }

    setup_kernel_smem_once();
    topk_transform_512_sglang_kernel<<<static_cast<unsigned int>(batch), kBlockSize, kSMEM, cuda_stream>>>(params);
}

} // namespace infinicore::op::deepseek_v4_topk_transform_512
