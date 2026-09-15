#pragma once

#include "../common/op.hpp"

#include <string>

namespace infinicore::op::deepseek_v4 {

void topk_(Tensor topk_weights,
           Tensor topk_indices,
           const Tensor &router_logits,
           const Tensor &correction_bias,
           bool renormalize,
           const std::string &scoring_func = "sqrtsoftplus");

void topk_aten_(Tensor topk_weights,
                Tensor topk_indices,
                const Tensor &router_logits,
                const Tensor &correction_bias,
                bool renormalize,
                const std::string &scoring_func = "sqrtsoftplus");

void topk_kernel_(Tensor topk_weights,
                  Tensor topk_indices,
                  const Tensor &router_logits,
                  const Tensor &correction_bias,
                  bool renormalize);

void topk_generic_kernel_(Tensor topk_weights,
                          Tensor topk_indices,
                          const Tensor &router_logits,
                          const Tensor &correction_bias,
                          bool renormalize);

} // namespace infinicore::op::deepseek_v4
