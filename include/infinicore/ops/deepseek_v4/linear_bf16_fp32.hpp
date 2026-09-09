#pragma once

#include "../common/op.hpp"

namespace infinicore::op::deepseek_v4 {

Tensor linear_bf16_fp32(const Tensor &x, const Tensor &weight);
void linear_bf16_fp32_(Tensor out, const Tensor &x, const Tensor &weight);

Tensor linear_bf16_fp32_aten(const Tensor &x, const Tensor &weight);
void linear_bf16_fp32_aten_(Tensor out, const Tensor &x, const Tensor &weight);

} // namespace infinicore::op::deepseek_v4
