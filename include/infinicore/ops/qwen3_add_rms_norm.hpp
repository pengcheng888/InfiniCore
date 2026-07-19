#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <utility>

namespace infinicore::op {

std::pair<Tensor, Tensor> qwen3_add_rms_norm(const Tensor &a, const Tensor &b, const Tensor &weight, float epsilon = 1e-6f);
void qwen3_add_rms_norm_(Tensor out, Tensor residual, const Tensor &a, const Tensor &b, const Tensor &weight, float epsilon = 1e-6f);
void qwen3_add_rms_norm_inplace(Tensor input, Tensor residual, const Tensor &weight, float epsilon = 1e-6f);

} // namespace infinicore::op
