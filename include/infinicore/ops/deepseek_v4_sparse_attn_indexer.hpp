#pragma once

#include "common/op.hpp"

#include <optional>

namespace infinicore::op {

void deepseek_v4_sparse_attn_indexer_prefill_(const Tensor &q,
                                              const Tensor &k,
                                              const Tensor &weights,
                                              const Tensor &cu_seqlen_ks,
                                              const Tensor &cu_seqlen_ke,
                                              Tensor logits,
                                              Tensor topk_indices,
                                              std::optional<Tensor> kv_scale,
                                              int topk_tokens,
                                              bool clean_logits);

void deepseek_v4_sparse_attn_indexer_decode_(const Tensor &q,
                                             const Tensor &fused_kv_cache,
                                             const Tensor &weights,
                                             const Tensor &context_lens,
                                             const Tensor &block_table,
                                             const Tensor &schedule_meta,
                                             Tensor logits,
                                             Tensor topk_indices,
                                             int max_context_len,
                                             int next_n,
                                             int topk_tokens,
                                             bool clean_logits);

} // namespace infinicore::op
