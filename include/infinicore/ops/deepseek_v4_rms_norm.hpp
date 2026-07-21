#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

Tensor deepseek_v4_rms_norm(const Tensor &x, const Tensor &weight, float epsilon);
void deepseek_v4_rms_norm_(Tensor y, const Tensor &x, const Tensor &weight, float epsilon);

} // namespace infinicore::op
