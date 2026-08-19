#include "infinicore/ops/deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang.hpp"

namespace infinicore::op {

void deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang_(const Tensor &q_input,
                                                             Tensor q_fp8,
                                                             const Tensor &weight,
                                                             Tensor weights_out,
                                                             float weight_scale,
                                                             const Tensor &freqs_cis,
                                                             const Tensor &positions) {
    deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang_kernel_(q_input,
                                                                   q_fp8,
                                                                   weight,
                                                                   weights_out,
                                                                   weight_scale,
                                                                   freqs_cis,
                                                                   positions);
}

} // namespace infinicore::op
