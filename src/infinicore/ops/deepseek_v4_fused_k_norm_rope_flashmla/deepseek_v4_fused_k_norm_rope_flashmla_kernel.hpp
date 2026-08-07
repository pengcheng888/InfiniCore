#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_fused_k_norm_rope_flashmla_native {

void launch_fused_k_norm_rope_flashmla(const void *kv,
                                       const void *kv_weight,
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

} // namespace infinicore::op::deepseek_v4_fused_k_norm_rope_flashmla_native
