#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

Tensor deepseek_v4_linear_bf16_fp32(const Tensor &x, const Tensor &weight);
void deepseek_v4_linear_bf16_fp32_(Tensor out, const Tensor &x, const Tensor &weight);

} // namespace infinicore::op
