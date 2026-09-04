#pragma once

#include "infinicore/ops/flash_mla/fwd_kvcache_mla.hpp"
#include "infinicore/tensor.hpp"

#include <cstdint>
#include <optional>

namespace infinicore::op::flash_mla::fwd_kvcache_mla_metax {

#if defined(ENABLE_ATEN) && defined(ENABLE_METAX_API)

void fwd_kvcache_mla_impl(
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

#endif

} // namespace infinicore::op::flash_mla::fwd_kvcache_mla_metax
