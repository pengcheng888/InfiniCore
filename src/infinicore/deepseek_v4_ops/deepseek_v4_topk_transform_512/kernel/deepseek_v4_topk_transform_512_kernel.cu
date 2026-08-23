#include "deepseek_v4_topk_transform_512_kernel.hpp"

#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

namespace infinicore::op::deepseek_v4_topk_transform_512 {
namespace {

constexpr int kTopK = 512;

__device__ __forceinline__ int64_t load_int_value(const void *__restrict__ ptr, int64_t idx, bool i64) {
    return i64 ? reinterpret_cast<const int64_t *>(ptr)[idx]
               : static_cast<int64_t>(reinterpret_cast<const int32_t *>(ptr)[idx]);
}

__device__ __forceinline__ int32_t transform_raw_index(int raw,
                                                       const void *__restrict__ page_table,
                                                       bool page_table_i64,
                                                       int64_t page_table_row_base,
                                                       int page_size) {
    if (raw < 0) {
        return -1;
    }
    const int64_t page_idx = raw / page_size;
    const int64_t offset = raw - page_idx * page_size;
    const int64_t physical_page = load_int_value(page_table, page_table_row_base + page_idx, page_table_i64);
    return static_cast<int32_t>(physical_page * page_size + offset);
}

__global__ void topk_transform_512_kernel(const float *__restrict__ scores,
                                          int64_t score_stride0,
                                          const void *__restrict__ seq_lens,
                                          bool seq_lens_i64,
                                          const void *__restrict__ page_table,
                                          bool page_table_i64,
                                          int64_t page_table_stride0,
                                          int32_t *__restrict__ out_page_indices,
                                          int64_t out_stride0,
                                          int64_t batch,
                                          int64_t max_seq_len,
                                          int page_size) {
    const int64_t row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= batch) {
        return;
    }

    int64_t seq_len = load_int_value(seq_lens, row, seq_lens_i64);
    if (seq_len < 0) {
        seq_len = 0;
    }
    if (seq_len > max_seq_len) {
        seq_len = max_seq_len;
    }
    const int64_t table_base = row * page_table_stride0;

    if (max_seq_len <= kTopK || seq_len <= kTopK) {
        for (int i = lane; i < kTopK; i += blockDim.x) {
            const int raw = i < seq_len ? i : -1;
            out_page_indices[row * out_stride0 + i] = transform_raw_index(raw, page_table, page_table_i64, table_base, page_size);
        }
        return;
    }

    if (lane != 0) {
        return;
    }

    int32_t *out = out_page_indices + row * out_stride0;
    const float *row_scores = scores + row * score_stride0;
    for (int rank = 0; rank < kTopK; ++rank) {
        float best = -INFINITY;
        int best_idx = -1;
        for (int pos = 0; pos < seq_len; ++pos) {
            bool used = false;
            for (int prev = 0; prev < rank; ++prev) {
                if (out[prev] == pos) {
                    used = true;
                    break;
                }
            }
            const float score = row_scores[pos];
            if (!used && score > best) {
                best = score;
                best_idx = pos;
            }
        }
        out[rank] = best_idx;
    }

    for (int rank = 0; rank < kTopK; ++rank) {
        out[rank] = transform_raw_index(out[rank], page_table, page_table_i64, table_base, page_size);
    }
}

} // namespace

void launch_topk_transform_512(const float *scores,
                               int64_t score_stride0,
                               const void *seq_lens,
                               bool seq_lens_i64,
                               const void *page_table,
                               bool page_table_i64,
                               int64_t page_table_stride0,
                               int32_t *out_page_indices,
                               int64_t out_stride0,
                               int64_t batch,
                               int64_t max_seq_len,
                               int page_size,
                               void *stream) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    topk_transform_512_kernel<<<static_cast<unsigned int>(batch), 256, 0, cuda_stream>>>(
        scores,
        score_stride0,
        seq_lens,
        seq_lens_i64,
        page_table,
        page_table_i64,
        page_table_stride0,
        out_page_indices,
        out_stride0,
        batch,
        max_seq_len,
        page_size);
}

} // namespace infinicore::op::deepseek_v4_topk_transform_512
