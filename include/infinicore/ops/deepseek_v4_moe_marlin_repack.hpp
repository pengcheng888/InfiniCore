#pragma once

#include "common/op.hpp"

namespace infinicore::op {

void deepseek_v4_moe_marlin_repack_(Tensor output, const Tensor &weight);

} // namespace infinicore::op
