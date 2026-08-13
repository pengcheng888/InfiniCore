#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

#include <optional>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4LmslimHipblasltChannelwiseLinearW8A8,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          std::optional<Tensor>,
                          Tensor,
                          Tensor,
                          const Tensor &);

void deepseek_v4_lmslim_hipblaslt_channelwise_linear_w8a8_(Tensor output,
                                                           const Tensor &input,
                                                           const Tensor &weight,
                                                           const Tensor &weight_scale,
                                                           std::optional<Tensor> bias,
                                                           Tensor q_input,
                                                           Tensor input_scale,
                                                           const Tensor &smooth_scale);

} // namespace infinicore::op
