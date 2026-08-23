#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

namespace deepseek_v4 {

INFINICORE_GRAPH_OP_CLASS(RmsnormSelf, Tensor, const Tensor &, float);

} // namespace deepseek_v4

Tensor deepseek_v4_rmsnorm_self_aten(const Tensor &x, float epsilon);
void deepseek_v4_rmsnorm_self_aten_(Tensor out, const Tensor &x, float epsilon);
Tensor deepseek_v4_rmsnorm_self_kernel(const Tensor &x, float epsilon);
void deepseek_v4_rmsnorm_self_kernel_(Tensor out, const Tensor &x, float epsilon);
Tensor deepseek_v4_rmsnorm_self(const Tensor &x, float epsilon);
void deepseek_v4_rmsnorm_self_(Tensor out, const Tensor &x, float epsilon);

} // namespace infinicore::op
