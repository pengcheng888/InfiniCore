#pragma once

#include "flash_mla_sched_meta/flash_mla_sched_meta.hpp"

#include "../../graph/graph.hpp"
#include "../common/op.hpp"

#include <cstdint>
#include <optional>

namespace infinicore::op::flash_mla {

using GetMlaMetadataImplSchema = void (*)(
    const Tensor &cache_seqlens,
    FlashMLASchedMeta &sched_meta,
    int64_t num_q_tokens_per_head_k,
    int64_t num_heads_k,
    std::optional<int64_t> num_heads_q,
    bool is_fp8_kvcache,
    std::optional<int64_t> topk);

common::OpDispatcher<GetMlaMetadataImplSchema> &get_mla_metadata_impl_dispatcher();

INFINICORE_GRAPH_OP_CLASS(GetMlaMetadata,
                          const Tensor &cache_seqlens,
                          FlashMLASchedMeta &sched_meta,
                          int64_t num_q_tokens_per_head_k,
                          int64_t num_heads_k,
                          std::optional<int64_t> num_heads_q,
                          bool is_fp8_kvcache,
                          std::optional<int64_t> topk);

void get_mla_metadata_(
    FlashMLASchedMeta &sched_meta,
    const Tensor &cache_seqlens,
    int64_t num_q_tokens_per_head_k,
    int64_t num_heads_k,
    std::optional<int64_t> num_heads_q = std::nullopt,
    bool is_fp8_kvcache = false,
    std::optional<int64_t> topk = std::nullopt);

} // namespace infinicore::op::flash_mla
