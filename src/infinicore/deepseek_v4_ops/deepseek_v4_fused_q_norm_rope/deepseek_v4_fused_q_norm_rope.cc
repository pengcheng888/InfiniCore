#include "infinicore/ops/deepseek_v4_fused_q_norm_rope.hpp"

namespace infinicore::op {

void deepseek_v4_fused_q_norm_rope_(Tensor q_out,
                                    const Tensor &q_input,
                                    float epsilon,
                                    const Tensor &freqs_cis,
                                    const Tensor &positions) {
    deepseek_v4_fused_q_norm_rope_kernel_(q_out, q_input, epsilon, freqs_cis, positions);
}

} // namespace infinicore::op
