#include "infinicore/ops/deepseek_v4_biased_topk.hpp"

namespace infinicore::op {

void deepseek_v4_topk_(Tensor topk_weights,
                       Tensor topk_indices,
                       const Tensor &router_logits,
                       const Tensor &correction_bias,
                       bool renormalize) {
    deepseek_v4_topk_kernel_(topk_weights,
                             topk_indices,
                             router_logits,
                             correction_bias,
                             renormalize);
}

} // namespace infinicore::op
