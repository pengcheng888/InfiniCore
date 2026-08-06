#pragma once

#include "infinicore/tensor.hpp"

namespace infinicore::op::deepseek_v4_linear_bf16_fp32_impl {

void check_shapes(const Tensor &out, const Tensor &x, const Tensor &weight, const char *op_name);

} // namespace infinicore::op::deepseek_v4_linear_bf16_fp32_impl
