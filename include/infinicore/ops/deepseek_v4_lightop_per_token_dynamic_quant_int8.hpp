#pragma once

#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4LightopPerTokenDynamicQuantInt8,
                          Tensor,
                          const Tensor &,
                          Tensor,
                          const Tensor &);

void deepseek_v4_lightop_per_token_dynamic_quant_int8_(Tensor q_input,
                                                       const Tensor &input,
                                                       Tensor input_scale,
                                                       const Tensor &smooth_scale);

} // namespace infinicore::op
