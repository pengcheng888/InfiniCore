#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4C4ActQuantFusedScaleKernel,
                          const Tensor &,
                          const Tensor &,
                          Tensor,
                          Tensor,
                          Tensor,
                          float);

void deepseek_v4_c4_act_quant_fused_scale_kernel_(const Tensor &q,
                                                  const Tensor &indexer_weights,
                                                  Tensor q_fp8,
                                                  Tensor q_scale,
                                                  Tensor fused_weights,
                                                  float weight_scale);

} // namespace infinicore::op
