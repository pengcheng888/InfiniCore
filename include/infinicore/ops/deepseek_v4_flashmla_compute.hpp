#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"
#include "deepseek_v4_compress_fused_norm_rope.hpp"
#include "deepseek_v4_compress_stateful.hpp"

#include <optional>

namespace infinicore::op {

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

INFINICORE_GRAPH_OP_CLASS(DeepseekV4FlashMlaSparseAttentionOutWorkspace,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          std::optional<Tensor>,
                          Tensor,
                          Tensor,
                          Tensor,
                          Tensor,
                          Tensor,
                          Tensor,
                          float,
                          int,
                          int,
                          std::optional<Tensor>,
                          std::optional<Tensor>,
                          std::optional<Tensor>,
                          int);

INFINICORE_GRAPH_OP_CLASS(DeepseekV4FlashMlaSparseAttentionMetadata,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          int,
                          std::optional<Tensor>,
                          int);

Tensor deepseek_v4_c4_compress_prefill_naive(const Tensor &kv_score_input,
                                                 const Tensor &ape);

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

void deepseek_v4_flashmla_sparse_attention_out_workspace_(
    const Tensor &q,
    const Tensor &raw_cache,
    const Tensor &indices,
    const Tensor &topk_lengths,
    std::optional<Tensor> attn_sink,
    Tensor output,
    Tensor lse,
    Tensor lse_accum,
    Tensor o_accum,
    Tensor tile_scheduler_metadata,
    Tensor num_splits,
    float softmax_scale,
    int page_size,
    int head_dim_v,
    std::optional<Tensor> extra_raw_cache = std::nullopt,
    std::optional<Tensor> extra_indices = std::nullopt,
    std::optional<Tensor> extra_topk_lengths = std::nullopt,
    int extra_page_size = 0);

void deepseek_v4_flashmla_sparse_attention_metadata_(Tensor tile_scheduler_metadata,
                                                     Tensor num_splits,
                                                     const Tensor &topk_lengths,
                                                     int topk,
                                                     std::optional<Tensor> extra_topk_lengths = std::nullopt,
                                                     int extra_topk = -1);

} // namespace infinicore::op
