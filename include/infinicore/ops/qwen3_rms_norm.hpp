#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

Tensor qwen3_rms_norm(const Tensor &x, const Tensor &weight, float epsilon = 1e-6f);
void qwen3_rms_norm_(Tensor y, const Tensor &x, const Tensor &weight, float epsilon = 1e-6f);

} // namespace infinicore::op
