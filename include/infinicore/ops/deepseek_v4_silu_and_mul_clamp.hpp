#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

Tensor deepseek_v4_silu_and_mul_clamp(const Tensor &x, float swiglu_limit);
void deepseek_v4_silu_and_mul_clamp_(Tensor out, const Tensor &x, float swiglu_limit);

} // namespace infinicore::op
