#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

std::pair<Tensor, Tensor> deepseek_v4_add_rms_norm(const Tensor &a, const Tensor &b, const Tensor &weight, float epsilon);
void deepseek_v4_add_rms_norm_(Tensor out, Tensor residual, const Tensor &a, const Tensor &b, const Tensor &weight, float epsilon);
void deepseek_v4_add_rms_norm_inplace(Tensor input, Tensor residual, const Tensor &weight, float epsilon);

} // namespace infinicore::op
