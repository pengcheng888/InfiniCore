#pragma once

#include "infinicore/ops/flash_mla/get_mla_decoding_metadata.hpp"
#include "infinicore/tensor.hpp"

#include <cstdint>
#include <optional>

namespace infinicore::op::flash_mla::get_mla_decoding_metadata_metax {

#if defined(ENABLE_ATEN) && defined(ENABLE_METAX_API)

void get_mla_decoding_metadata_impl(
    Tensor &tile_scheduler_metadata,
    Tensor &num_splits,
    const Tensor &cache_seqlens,
    int64_t num_q_tokens_per_head_k,
    int64_t num_heads_k,
    std::optional<int64_t> num_heads_q,
    bool is_fp8_kvcache,
    std::optional<int64_t> topk);

#endif

} // namespace infinicore::op::flash_mla::get_mla_decoding_metadata_metax
