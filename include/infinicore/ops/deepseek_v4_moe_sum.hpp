#pragma once

#include "common/op.hpp"

namespace infinicore::op {

void deepseek_v4_moe_sum_(Tensor output, const Tensor &input);

} // namespace infinicore::op
