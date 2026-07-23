#pragma once

#include "common/op.hpp"

namespace infinicore::op {

Tensor deepseek_v4_rmsnorm_self(const Tensor &x, float epsilon);
void deepseek_v4_rmsnorm_self_(Tensor out, const Tensor &x, float epsilon);

} // namespace infinicore::op

