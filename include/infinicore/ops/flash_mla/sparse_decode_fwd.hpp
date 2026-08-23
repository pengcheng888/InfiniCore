#pragma once

#include "../../device.hpp"
#include "../../graph/graph.hpp"
#include "../common/op.hpp"

#include <optional>
#include <tuple>

namespace infinicore::op {

namespace flash_mla {

INFINICORE_GRAPH_OP_CLASS(SparseDecodeFwd,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          std::optional<Tensor>,
                          std::optional<Tensor>,
                          std::optional<Tensor>,
                          std::optional<Tensor>,
                          std::optional<Tensor>,
                          int64_t,
                          double);

std::tuple<Tensor, Tensor, Tensor, Tensor> sparse_decode_fwd(
    const Tensor &q,
    const Tensor &k_cache,
    const Tensor &indices,
    std::optional<Tensor> topk_length,
    std::optional<Tensor> attn_sink,
    std::optional<Tensor> tile_scheduler_metadata,
    std::optional<Tensor> num_splits,
    std::optional<Tensor> extra_k_cache,
    std::optional<Tensor> extra_indices_in_kvcache,
    std::optional<Tensor> extra_topk_length,
    int64_t head_dim_v,
    double softmax_scale);

void sparse_decode_fwd_(
    Tensor &out,
    Tensor &lse,
    Tensor &new_tile_scheduler_metadata,
    Tensor &new_num_splits,
    const Tensor &q,
    const Tensor &k_cache,
    const Tensor &indices,
    std::optional<Tensor> topk_length,
    std::optional<Tensor> attn_sink,
    std::optional<Tensor> tile_scheduler_metadata,
    std::optional<Tensor> num_splits,
    std::optional<Tensor> extra_k_cache,
    std::optional<Tensor> extra_indices_in_kvcache,
    std::optional<Tensor> extra_topk_length,
    int64_t head_dim_v,
    double softmax_scale);

} // namespace flash_mla

} // namespace infinicore::op
