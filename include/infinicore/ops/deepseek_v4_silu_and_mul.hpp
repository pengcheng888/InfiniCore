#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

Tensor deepseek_v4_silu_and_mul(const Tensor &x);
void deepseek_v4_silu_and_mul_(Tensor out, const Tensor &x);

} // namespace infinicore::op
