#pragma once

#include "flash_mla_sched_meta/flash_mla_sched_meta.hpp"

#include "../../device.hpp"
#include "../../graph/graph.hpp"
#include "../common/op.hpp"

#include <cstdint>
#include <optional>
#include <utility>

namespace infinicore::op {

namespace flash_mla {

using FlashMlaWithKvcacheImplSchema = void (*)(
    Tensor &out,
    Tensor &lse,
    const Tensor &q,
    const Tensor &k_cache,
    std::optional<Tensor> block_table,
    std::optional<Tensor> cache_seqlens,
    int64_t head_dim_v,
    const FlashMLASchedMeta &tile_scheduler_metadata,
    std::optional<Tensor> num_splits,
    std::optional<double> softmax_scale,
    bool causal,
    bool is_fp8_kvcache,
    std::optional<Tensor> indices,
    std::optional<Tensor> attn_sink,
    std::optional<Tensor> extra_k_cache,
    std::optional<Tensor> extra_indices_in_kvcache,
    std::optional<Tensor> topk_length,
    std::optional<Tensor> extra_topk_length);

common::OpDispatcher<FlashMlaWithKvcacheImplSchema> &flash_mla_with_kvcache_impl_dispatcher();

INFINICORE_GRAPH_OP_CLASS(FlashMlaWithKvcache,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          std::optional<Tensor>,
                          std::optional<Tensor>,
                          int64_t,
                          const FlashMLASchedMeta &,
                          std::optional<Tensor>,
                          std::optional<double>,
                          bool,
                          bool,
                          std::optional<Tensor>,
                          std::optional<Tensor>,
                          std::optional<Tensor>,
                          std::optional<Tensor>,
                          std::optional<Tensor>,
                          std::optional<Tensor>);

std::pair<Tensor, Tensor> flash_mla_with_kvcache(
    const Tensor &q,
    const Tensor &k_cache,
    std::optional<Tensor> block_table,
    std::optional<Tensor> cache_seqlens,
    int64_t head_dim_v,
    const FlashMLASchedMeta &tile_scheduler_metadata,
    std::optional<Tensor> num_splits = std::nullopt,
    std::optional<double> softmax_scale = std::nullopt,
    bool causal = false,
    bool is_fp8_kvcache = false,
    std::optional<Tensor> indices = std::nullopt,
    std::optional<Tensor> attn_sink = std::nullopt,
    std::optional<Tensor> extra_k_cache = std::nullopt,
    std::optional<Tensor> extra_indices_in_kvcache = std::nullopt,
    std::optional<Tensor> topk_length = std::nullopt,
    std::optional<Tensor> extra_topk_length = std::nullopt);

} // namespace flash_mla

} // namespace infinicore::op
