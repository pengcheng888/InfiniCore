#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_topk_transform_512 {

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
                                      void *stream);

} // namespace infinicore::op::deepseek_v4_topk_transform_512
