#pragma once

#include "../../device.hpp"
#include "../../graph/graph.hpp"
#include "../common/op.hpp"

#include <cstdint>
#include <optional>
#include <utility>

namespace infinicore::op {

namespace flash_mla {

using GetMlaDecodingMetadataImplSchema = void (*)(
    Tensor &tile_scheduler_metadata,
    Tensor &num_splits,
    const Tensor &cache_seqlens,
    int64_t num_q_tokens_per_head_k,
    int64_t num_heads_k,
    std::optional<int64_t> num_heads_q,
    bool is_fp8_kvcache,
    std::optional<int64_t> topk);

common::OpDispatcher<GetMlaDecodingMetadataImplSchema> &get_mla_decoding_metadata_impl_dispatcher();

INFINICORE_GRAPH_OP_CLASS(GetMlaDecodingMetadata,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          int64_t,
                          int64_t,
                          std::optional<int64_t>,
                          bool,
                          std::optional<int64_t>);

std::pair<Tensor, Tensor> get_mla_decoding_metadata(
    Tensor tile_scheduler_metadata,
    Tensor num_splits,
    const Tensor &cache_seqlens,
    int64_t num_q_tokens_per_head_k,
    int64_t num_heads_k,
    std::optional<int64_t> num_heads_q,
    bool is_fp8_kvcache,
    std::optional<int64_t> topk);

} // namespace flash_mla

} // namespace infinicore::op
