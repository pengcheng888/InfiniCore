#pragma once

#include "../../device.hpp"
#include "../../graph/graph.hpp"
#include "../common/op.hpp"

#include <optional>

namespace infinicore::op {

namespace flash_mla {

using DenseDecodeFwdImplSchema = void (*)(
    Tensor &out,
    Tensor &lse,
    std::optional<Tensor> &new_tile_scheduler_metadata,
    std::optional<Tensor> &new_num_splits,
    const Tensor &q,
    const Tensor &k_cache,
    int64_t head_dim_v,
    const Tensor &cache_seqlens,
    const Tensor &block_table,
    double softmax_scale,
    bool causal,
    std::optional<Tensor> tile_scheduler_metadata,
    std::optional<Tensor> num_splits);

common::OpDispatcher<DenseDecodeFwdImplSchema> &dense_decode_fwd_impl_dispatcher();

INFINICORE_GRAPH_OP_CLASS(DenseDecodeFwd,
                          Tensor,
                          Tensor,
                          std::optional<Tensor>,
                          std::optional<Tensor>,
                          const Tensor &,
                          const Tensor &,
                          int64_t,
                          const Tensor &,
                          const Tensor &,
                          double,
                          bool,
                          std::optional<Tensor>,
                          std::optional<Tensor>);

void dense_decode_fwd_(
    Tensor &out,
    Tensor &lse,
    std::optional<Tensor> &new_tile_scheduler_metadata,
    std::optional<Tensor> &new_num_splits,
    const Tensor &q,
    const Tensor &k_cache,
    int64_t head_dim_v,
    const Tensor &cache_seqlens,
    const Tensor &block_table,
    double softmax_scale,
    bool causal,
    std::optional<Tensor> tile_scheduler_metadata,
    std::optional<Tensor> num_splits);

} // namespace flash_mla

} // namespace infinicore::op
