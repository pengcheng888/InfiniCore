#include "infinicore/ops/deepseek_v4_moe_w8a8.hpp"

namespace infinicore::op {

void deepseek_v4_moe_w8a8_(Tensor y,
                           const Tensor &x,
                           const Tensor &topk_weights,
                           const Tensor &topk_indices,
                           const Tensor &w13,
                           const Tensor &w13_scale,
                           const Tensor &w2,
                           const Tensor &w2_scale,
                           double swiglu_limit) {
    deepseek_v4_moe_w8a8_naive_(y, x, topk_weights, topk_indices, w13, w13_scale, w2, w2_scale, swiglu_limit);
}

} // namespace infinicore::op
