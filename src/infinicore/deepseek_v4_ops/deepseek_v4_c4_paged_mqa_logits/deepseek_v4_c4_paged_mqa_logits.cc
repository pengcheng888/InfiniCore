#include "infinicore/ops/deepseek_v4_c4_paged_mqa_logits.hpp"

namespace infinicore::op {

void deepseek_v4_c4_paged_mqa_logits_(const Tensor &q_fp8,
                                      const Tensor &fused_weights,
                                      const Tensor &indexer_kv_cache_raw,
                                      const Tensor &c4_seq_lens,
                                      const Tensor &page_table,
                                      Tensor logits,
                                      int max_c4_seq_len,
                                      int page_size,
                                      bool clean_logits) {
    deepseek_v4_c4_paged_mqa_logits_lightop_(q_fp8,
                                             fused_weights,
                                             indexer_kv_cache_raw,
                                             c4_seq_lens,
                                             page_table,
                                             logits,
                                             max_c4_seq_len,
                                             page_size,
                                             clean_logits);
}

} // namespace infinicore::op
