#pragma once

#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

void deepseek_v4_rms_norm_dynamic_per_token_quant_(Tensor result,
                                                   const Tensor &input,
                                                   const Tensor &weight,
                                                   Tensor scale,
                                                   float epsilon,
                                                   std::optional<Tensor> scale_ub,
                                                   std::optional<Tensor> residual);

} // namespace infinicore::op
