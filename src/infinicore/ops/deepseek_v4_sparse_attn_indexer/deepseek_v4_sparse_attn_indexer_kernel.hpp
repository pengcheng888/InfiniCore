#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_sparse_attn_indexer {

enum Dsv4ScalarType : int {
    kDsv4BF16 = 0,
    kDsv4F16 = 1,
    kDsv4F32 = 2,
};

void launch_c4_act_quant_fused_scale(const void *q,
                                     int q_dtype,
                                     const void *weights,
                                     int weights_dtype,
                                     uint8_t *q_fp8,
                                     float *q_scale,
                                     float *fused_weights,
                                     int64_t rows,
                                     float weight_scale,
                                     void *stream);

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
                               void *stream);

} // namespace infinicore::op::deepseek_v4_sparse_attn_indexer
