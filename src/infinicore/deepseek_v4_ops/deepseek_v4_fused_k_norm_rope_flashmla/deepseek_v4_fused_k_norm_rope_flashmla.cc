#include "infinicore/ops/deepseek_v4_fused_k_norm_rope_flashmla.hpp"

namespace infinicore::op {

void deepseek_v4_fused_k_norm_rope_flashmla_(const Tensor &kv,
                                             const Tensor &kv_weight,
                                             float epsilon,
                                             const Tensor &freqs_cis,
                                             const Tensor &positions,
                                             const Tensor &out_loc,
                                             Tensor kvcache,
                                             int page_size) {
    deepseek_v4_fused_k_norm_rope_flashmla_kernel_(kv,
                                                   kv_weight,
                                                   epsilon,
                                                   freqs_cis,
                                                   positions,
                                                   out_loc,
                                                   kvcache,
                                                   page_size);
}

} // namespace infinicore::op
