#pragma once

#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

void deepseek_v4_moe_topk_sigmoid_(Tensor topk_weights,
                                   Tensor topk_indices,
                                   const Tensor &gating_output,
                                   bool renormalize,
                                   std::optional<Tensor> correction_bias);

} // namespace infinicore::op
