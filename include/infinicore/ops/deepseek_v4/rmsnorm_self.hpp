#pragma once

#include "../common/op.hpp"

namespace infinicore::op::deepseek_v4 {

Tensor rmsnorm_self(const Tensor &x, float epsilon);
void rmsnorm_self_(Tensor out, const Tensor &x, float epsilon);

Tensor rmsnorm_self_aten(const Tensor &x, float epsilon);
void rmsnorm_self_aten_(Tensor out, const Tensor &x, float epsilon);

} // namespace infinicore::op::deepseek_v4
