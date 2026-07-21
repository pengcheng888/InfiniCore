#pragma once

#include "../device.hpp"
#include "common/op.hpp"

#include <optional>

namespace infinicore::op {

void deepseek_v4_moe_topk_softmax_(Tensor topk_weights,
                                   Tensor topk_indices,
                                   const Tensor &gating_output,
                                   bool renormalize,
                                   float moe_softcapping,
                                   std::optional<Tensor> correction_bias);

} // namespace infinicore::op
