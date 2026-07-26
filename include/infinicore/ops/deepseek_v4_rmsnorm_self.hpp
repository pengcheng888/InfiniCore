#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4RmsnormSelfKernel, Tensor, const Tensor &, float);

Tensor deepseek_v4_rmsnorm_self_naive(const Tensor &x, float epsilon);
void deepseek_v4_rmsnorm_self_naive_(Tensor out, const Tensor &x, float epsilon);
Tensor deepseek_v4_rmsnorm_self_kernel(const Tensor &x, float epsilon);
void deepseek_v4_rmsnorm_self_kernel_(Tensor out, const Tensor &x, float epsilon);
Tensor deepseek_v4_rmsnorm_self(const Tensor &x, float epsilon);
void deepseek_v4_rmsnorm_self_(Tensor out, const Tensor &x, float epsilon);

} // namespace infinicore::op
