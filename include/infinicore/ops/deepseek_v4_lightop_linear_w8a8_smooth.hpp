#pragma once

#include "../graph/graph.hpp"
#include "common/op.hpp"

#include <optional>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4LightopLinearW8A8Smooth,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const std::optional<Tensor> &,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          bool,
                          int);

void deepseek_v4_lightop_linear_w8a8_smooth_(Tensor output,
                                             const Tensor &input,
                                             const Tensor &weight,
                                             const Tensor &weight_scale,
                                             const std::optional<Tensor> &bias,
                                             Tensor q_input,
                                             Tensor input_scale,
                                             const Tensor &smooth_scale,
                                             bool is_tuned_slide_block,
                                             int tuned_slide_block);

void deepseek_v4_lightop_linear_w8a8_smooth_(Tensor output,
                                             const Tensor &input,
                                             const Tensor &weight,
                                             const Tensor &weight_scale,
                                             const std::optional<Tensor> &bias,
                                             Tensor q_input,
                                             Tensor input_scale,
                                             const Tensor &smooth_scale);

} // namespace infinicore::op
