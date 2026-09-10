#pragma once

#include "../common/op.hpp"

namespace infinicore::op::deepseek_v4 {

void moe_w8a8_(Tensor y,
               const Tensor &x,
               const Tensor &topk_weights,
               const Tensor &topk_indices,
               const Tensor &w13,
               const Tensor &w13_scale,
               const Tensor &w2,
               const Tensor &w2_scale,
               double swiglu_limit);

void moe_w8a8_aten_(Tensor y,
                    const Tensor &x,
                    const Tensor &topk_weights,
                    const Tensor &topk_indices,
                    const Tensor &w13,
                    const Tensor &w13_scale,
                    const Tensor &w2,
                    const Tensor &w2_scale,
                    double swiglu_limit);

} // namespace infinicore::op::deepseek_v4
