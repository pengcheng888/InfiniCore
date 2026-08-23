#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

namespace deepseek_v4 {

INFINICORE_GRAPH_OP_CLASS(FusedQNormRope,
                          Tensor,
                          const Tensor &,
                          float,
                          const Tensor &,
                          const Tensor &);

} // namespace deepseek_v4

void deepseek_v4_fused_q_norm_rope_aten_(Tensor q_out,
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
