#pragma once

#include "common/op.hpp"

namespace infinicore::op {

void deepseek_v4_paged_mqa_logits_metadata_(const Tensor &context_lens,
                                            Tensor schedule_meta,
                                            int block_kv,
                                            int num_sms);

void deepseek_v4_paged_mqa_logits_(const Tensor &q,
                                   const Tensor &fused_kv_cache,
                                   const Tensor &weights,
                                   const Tensor &context_lens,
                                   const Tensor &block_table,
                                   const Tensor &schedule_meta,
                                   Tensor logits,
                                   int max_context_len,
                                   bool clean_logits);

} // namespace infinicore::op
