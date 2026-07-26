#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

#include <optional>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4FusedRopeKernel, Tensor, std::optional<Tensor>, const Tensor &, const Tensor &, bool);

void deepseek_v4_fused_rope_(Tensor query,
                             std::optional<Tensor> key,
                             const Tensor &freqs_cis,
                             const Tensor &positions,
                             bool inverse);

} // namespace infinicore::op
