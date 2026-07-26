#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

#include <optional>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4CompressFusedNormRopeKernel,
                          Tensor,
                          const Tensor &,
                          float,
                          const Tensor &,
                          const Tensor &);
INFINICORE_GRAPH_OP_CLASS(DeepseekV4C4CompressStatefulKernel,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &);
INFINICORE_GRAPH_OP_CLASS(DeepseekV4C128CompressStatefulKernel,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          Tensor,
                          const Tensor &,
                          const Tensor &);

INFINICORE_GRAPH_OP_CLASS(DeepseekV4FlashMlaSparseAttentionWithMetadata,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          std::optional<Tensor>,
                          Tensor,
                          std::optional<Tensor>,
                          std::optional<Tensor>,
                          float,
                          int,
                          int,
                          std::optional<Tensor>,
                          std::optional<Tensor>,
                          std::optional<Tensor>,
                          int);

Tensor deepseek_v4_c4_compress_prefill_naive(const Tensor &kv_score_input,
                                                 const Tensor &ape);

Tensor deepseek_v4_c4_compress_stateful_naive(const Tensor &kv_score_input,
                                                  const Tensor &ape,
                                                  Tensor compressor_state,
                                                  const Tensor &write_loc,
                                                  const Tensor &extra_loc,
                                                  const Tensor &positions);

Tensor deepseek_v4_c4_compress_stateful_kernel(const Tensor &kv_score_input,
                                               const Tensor &ape,
                                               Tensor compressor_state,
                                               const Tensor &write_loc,
                                               const Tensor &extra_loc,
                                               const Tensor &positions);

Tensor deepseek_v4_c4_compress_stateful(const Tensor &kv_score_input,
                                        const Tensor &ape,
                                        Tensor compressor_state,
                                        const Tensor &write_loc,
                                        const Tensor &extra_loc,
                                        const Tensor &positions);

Tensor deepseek_v4_c128_compress_stateful_naive(const Tensor &kv_score_input,
                                                    const Tensor &ape,
                                                    Tensor compressor_state,
                                                    const Tensor &write_loc,
                                                    const Tensor &positions);

Tensor deepseek_v4_c128_compress_stateful_kernel(const Tensor &kv_score_input,
                                                 const Tensor &ape,
                                                 Tensor compressor_state,
                                                 const Tensor &write_loc,
                                                 const Tensor &positions);

Tensor deepseek_v4_c128_compress_stateful(const Tensor &kv_score_input,
                                          const Tensor &ape,
                                          Tensor compressor_state,
                                          const Tensor &write_loc,
                                          const Tensor &positions);

void deepseek_v4_compress_fused_norm_rope_naive_(Tensor input,
                                                     const Tensor &norm_weight,
                                                     float epsilon,
                                                     const Tensor &freqs_cis,
                                                     const Tensor &positions);

void deepseek_v4_compress_fused_norm_rope_kernel_(Tensor input,
                                                  const Tensor &norm_weight,
                                                  float epsilon,
                                                  const Tensor &freqs_cis,
                                                  const Tensor &positions);

void deepseek_v4_compress_fused_norm_rope_(Tensor input,
                                           const Tensor &norm_weight,
                                           float epsilon,
                                           const Tensor &freqs_cis,
                                           const Tensor &positions);

struct DeepseekV4FlashMLASparseAttentionSchedule {
    Tensor tile_scheduler_metadata;
    Tensor num_splits;
};

void deepseek_v4_flashmla_sparse_attention_(const Tensor &q,
                                            const Tensor &raw_cache,
                                            const Tensor &indices,
                                            const Tensor &topk_lengths,
                                            std::optional<Tensor> attn_sink,
                                            Tensor output,
                                            float softmax_scale,
                                            int page_size,
                                            int head_dim_v,
                                            std::optional<Tensor> extra_raw_cache = std::nullopt,
                                            std::optional<Tensor> extra_indices = std::nullopt,
                                            std::optional<Tensor> extra_topk_lengths = std::nullopt,
                                            int extra_page_size = 0);

DeepseekV4FlashMLASparseAttentionSchedule deepseek_v4_flashmla_sparse_attention_with_metadata_(
    const Tensor &q,
    const Tensor &raw_cache,
    const Tensor &indices,
    const Tensor &topk_lengths,
    std::optional<Tensor> attn_sink,
    Tensor output,
    std::optional<Tensor> tile_scheduler_metadata,
    std::optional<Tensor> num_splits,
    float softmax_scale,
    int page_size,
    int head_dim_v,
    std::optional<Tensor> extra_raw_cache = std::nullopt,
    std::optional<Tensor> extra_indices = std::nullopt,
    std::optional<Tensor> extra_topk_lengths = std::nullopt,
    int extra_page_size = 0);

} // namespace infinicore::op
