#pragma once

#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4LightopLinearW8A8Asm,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &);

INFINICORE_GRAPH_OP_CLASS(DeepseekV4LightopLinearW8A8AsmPerChannel,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          Tensor,
                          Tensor,
                          Tensor,
                          Tensor,
                          const Tensor &);

void deepseek_v4_lightop_linear_w8a8_asm_(Tensor output,
                                          const Tensor &q_input,
                                          const Tensor &weight,
                                          const Tensor &input_block_scale,
                                          const Tensor &weight_block_scale);

void deepseek_v4_lightop_linear_w8a8_asm_per_channel_(Tensor output,
                                                      const Tensor &input,
                                                      const Tensor &weight,
                                                      const Tensor &weight_scale,
                                                      Tensor q_input,
                                                      Tensor input_scale,
                                                      Tensor input_block_scale,
                                                      Tensor weight_block_scale,
                                                      const Tensor &smooth_scale);

} // namespace infinicore::op
