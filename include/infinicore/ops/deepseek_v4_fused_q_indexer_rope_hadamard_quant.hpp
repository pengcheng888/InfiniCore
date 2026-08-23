#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4FusedQIndexerRopeHadamardQuantKernel,
                          const Tensor &,
                          const Tensor &,
                          Tensor,
                          Tensor,
                          Tensor,
                          float,
                          const Tensor &,
                          const Tensor &);

void deepseek_v4_fused_q_indexer_rope_hadamard_quant_(const Tensor &q,
                                                      const Tensor &indexer_weights,
                                                      Tensor q_fp8,
                                                      Tensor q_scale,
                                                      Tensor fused_weights,
                                                      float weight_scale,
                                                      const Tensor &freqs_cis,
                                                      const Tensor &positions);

void deepseek_v4_fused_q_indexer_rope_hadamard_quant_kernel_(const Tensor &q,
                                                             const Tensor &indexer_weights,
                                                             Tensor q_fp8,
                                                             Tensor q_scale,
                                                             Tensor fused_weights,
                                                             float weight_scale,
                                                             const Tensor &freqs_cis,
                                                             const Tensor &positions);

} // namespace infinicore::op
