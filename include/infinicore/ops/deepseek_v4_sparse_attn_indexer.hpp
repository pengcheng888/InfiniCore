#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

#include <optional>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4C4SparseAttnIndexerNoLogits,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          Tensor,
                          int,
                          int,
                          float,
                          bool);

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

void deepseek_v4_c4_sparse_attn_indexer_(const Tensor &q,
                                         const Tensor &indexer_weights,
                                         const Tensor &indexer_kv_cache_raw,
                                         const Tensor &c4_seq_lens,
                                         const Tensor &page_table,
                                         Tensor logits,
                                         Tensor out_page_indices,
                                         int max_c4_seq_len,
                                         int page_size,
                                         float weight_scale,
                                         bool clean_logits);

void deepseek_v4_c4_sparse_attn_indexer_no_logits_(const Tensor &q,
                                                   const Tensor &indexer_weights,
                                                   const Tensor &indexer_kv_cache_raw,
                                                   const Tensor &c4_seq_lens,
                                                   const Tensor &page_table,
                                                   Tensor out_page_indices,
                                                   int max_c4_seq_len,
                                                   int page_size,
                                                   float weight_scale,
                                                   bool clean_logits);

void deepseek_v4_c4_act_quant_fused_scale_kernel_(const Tensor &q,
                                                  const Tensor &indexer_weights,
                                                  Tensor q_fp8,
                                                  Tensor q_scale,
                                                  Tensor fused_weights,
                                                  float weight_scale);

void deepseek_v4_topk_transform_512_kernel_(const Tensor &scores,
                                            const Tensor &seq_lens,
                                            const Tensor &page_table,
                                            Tensor out_page_indices,
                                            int page_size);

} // namespace infinicore::op
