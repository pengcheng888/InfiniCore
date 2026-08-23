#pragma once

#include <stdint.h>

namespace infinicore::op::deepseek_v4_indexer_compress_norm_rope_store {

void launch_indexer_compress_norm_rope_store(const void *kv,
                                             const void *norm_weight,
                                             int norm_weight_dtype,
                                             const float *freqs_cis,
                                             const void *positions,
                                             bool positions_i64,
                                             const void *out_loc,
                                             bool out_loc_i64,
                                             uint8_t *kvcache,
                                             int64_t tokens,
                                             int64_t kv_stride_batch,
                                             int page_size,
                                             int64_t page_bytes,
                                             float epsilon,
                                             void *stream);

} // namespace infinicore::op::deepseek_v4_indexer_compress_norm_rope_store
