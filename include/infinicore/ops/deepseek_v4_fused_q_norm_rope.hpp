#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4FusedQNormRopeKernel,
                          Tensor,
                          const Tensor &,
                          float,
                          const Tensor &,
                          const Tensor &);

void deepseek_v4_fused_q_norm_rope_naive_(Tensor q_out,
                                          const Tensor &q_input,
                                          float epsilon,
                                          const Tensor &freqs_cis,
                                          const Tensor &positions);

void deepseek_v4_fused_q_norm_rope_kernel_(Tensor q_out,
                                           const Tensor &q_input,
                                           float epsilon,
                                           const Tensor &freqs_cis,
                                           const Tensor &positions);

void deepseek_v4_fused_q_norm_rope_(Tensor q_out,
                                    const Tensor &q_input,
                                    float epsilon,
                                    const Tensor &freqs_cis,
                                    const Tensor &positions);

} // namespace infinicore::op
