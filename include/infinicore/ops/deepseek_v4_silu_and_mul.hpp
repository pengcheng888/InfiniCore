#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

namespace deepseek_v4 {

INFINICORE_GRAPH_OP_CLASS(SiluAndMul, Tensor, const Tensor &);

} // namespace deepseek_v4

Tensor deepseek_v4_silu_and_mul(const Tensor &x);
void deepseek_v4_silu_and_mul_(Tensor out, const Tensor &x);
void deepseek_v4_silu_and_mul_kernel_(Tensor out, const Tensor &x);
void deepseek_v4_silu_and_mul_dispatcher_(Tensor out, const Tensor &x);

} // namespace infinicore::op
