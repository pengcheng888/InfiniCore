#pragma once

#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4ExpandHcKernel, Tensor, const Tensor &, int64_t);

void deepseek_v4_expand_hc_(Tensor output, const Tensor &input, int64_t hc);

} // namespace infinicore::op
