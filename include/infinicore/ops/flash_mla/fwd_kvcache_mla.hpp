#pragma once

#include "../../device.hpp"
#include "../../graph/graph.hpp"
#include "../common/op.hpp"

#include <cstdint>
#include <optional>

namespace infinicore::op {

namespace flash_mla {

using FwdKvcacheMlaImplSchema = void (*)(
    Tensor &out,
    Tensor &lse,
    const Tensor &q,
    const Tensor &k_cache,
    std::optional<Tensor> k_cache_scale,
    int64_t head_dim_v,
    const Tensor &cache_seqlens,
    const Tensor &block_table,
    double softmax_scale,
    bool causal,
    const Tensor &tile_scheduler_metadata,
    const Tensor &num_splits,
    bool is_fp8_kvcache,
    std::optional<Tensor> extra_k_cache,
    std::optional<Tensor> extra_block_table,
    int64_t cp_world_size,
    int64_t cp_rank,
    std::optional<Tensor> cp_tot_seqused_k);

common::OpDispatcher<FwdKvcacheMlaImplSchema> &fwd_kvcache_mla_impl_dispatcher();

INFINICORE_GRAPH_OP_CLASS(FwdKvcacheMla,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          std::optional<Tensor>,
                          int64_t,
                          const Tensor &,
                          const Tensor &,
                          double,
                          bool,
                          const Tensor &,
                          const Tensor &,
                          bool,
                          std::optional<Tensor>,
                          std::optional<Tensor>,
                          int64_t,
                          int64_t,
                          std::optional<Tensor>);

void fwd_kvcache_mla_(
    Tensor &out,
    Tensor &lse,
    const Tensor &q,
    const Tensor &k_cache,
    std::optional<Tensor> k_cache_scale,
    int64_t head_dim_v,
    const Tensor &cache_seqlens,
    const Tensor &block_table,
    double softmax_scale,
    bool causal,
    const Tensor &tile_scheduler_metadata,
    const Tensor &num_splits,
    bool is_fp8_kvcache,
    std::optional<Tensor> extra_k_cache,
    std::optional<Tensor> extra_block_table,
    int64_t cp_world_size,
    int64_t cp_rank,
    std::optional<Tensor> cp_tot_seqused_k);

} // namespace flash_mla

} // namespace infinicore::op
