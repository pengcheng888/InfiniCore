#pragma once

#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

void deepseek_v4_static_scaled_int8_quant_(Tensor result,
                                           const Tensor &input,
                                           const Tensor &scale,
                                           std::optional<Tensor> azp);

} // namespace infinicore::op
