#pragma once

#include "common/op.hpp"

namespace infinicore::op {

void deepseek_v4_moe_w8a8_(Tensor y,
                           const Tensor &x,
                           const Tensor &topk_weights,
                           const Tensor &topk_indices,
                           const Tensor &w13,
                           const Tensor &w13_scale,
                           const Tensor &w2,
                           const Tensor &w2_scale,
                           double swiglu_limit);

void deepseek_v4_moe_w8a8_naive_(Tensor y,
                                 const Tensor &x,
                                 const Tensor &topk_weights,
                                 const Tensor &topk_indices,
                                 const Tensor &w13,
                                 const Tensor &w13_scale,
                                 const Tensor &w2,
                                 const Tensor &w2_scale,
                                 double swiglu_limit);

} // namespace infinicore::op
