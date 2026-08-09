#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4FusedQIndexerRopeHadamardQuantSglangKernel,
                          const Tensor &,
                          Tensor,
                          const Tensor &,
                          Tensor,
                          float,
                          const Tensor &,
                          const Tensor &);

void deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang_(const Tensor &q_input,
                                                             Tensor q_fp8,
                                                             const Tensor &weight,
                                                             Tensor weights_out,
                                                             float weight_scale,
                                                             const Tensor &freqs_cis,
                                                             const Tensor &positions);

void deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang_kernel_(const Tensor &q_input,
                                                                    Tensor q_fp8,
                                                                    const Tensor &weight,
                                                                    Tensor weights_out,
                                                                    float weight_scale,
                                                                    const Tensor &freqs_cis,
                                                                    const Tensor &positions);

} // namespace infinicore::op
