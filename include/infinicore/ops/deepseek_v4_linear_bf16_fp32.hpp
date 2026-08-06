#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4LinearBf16Fp32Kernel, Tensor, const Tensor &, const Tensor &);

Tensor deepseek_v4_linear_bf16_fp32(const Tensor &x, const Tensor &weight);
void deepseek_v4_linear_bf16_fp32_(Tensor out, const Tensor &x, const Tensor &weight);

Tensor deepseek_v4_linear_bf16_fp32_naive(const Tensor &x, const Tensor &weight);
void deepseek_v4_linear_bf16_fp32_naive_(Tensor out, const Tensor &x, const Tensor &weight);

Tensor deepseek_v4_linear_bf16_fp32_kernel(const Tensor &x, const Tensor &weight);
void deepseek_v4_linear_bf16_fp32_kernel_(Tensor out, const Tensor &x, const Tensor &weight);

Tensor deepseek_v4_linear_bf16_fp32_blas(const Tensor &x, const Tensor &weight);
void deepseek_v4_linear_bf16_fp32_blas_(Tensor out, const Tensor &x, const Tensor &weight);

} // namespace infinicore::op
