#pragma once

#include "common/op.hpp"

namespace infinicore::op {

void deepseek_v4_topk_naive_(Tensor topk_weights,
                              Tensor topk_indices,
                              const Tensor &router_logits,
                              const Tensor &correction_bias,
                              bool renormalize);

} // namespace infinicore::op
