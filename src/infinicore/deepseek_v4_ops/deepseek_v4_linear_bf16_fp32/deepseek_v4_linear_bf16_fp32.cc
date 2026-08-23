#include "infinicore/ops/deepseek_v4_linear_bf16_fp32.hpp"

#include "infinicore/dtype.hpp"

#include <stdexcept>

namespace infinicore::op {

Tensor deepseek_v4_linear_bf16_fp32(const Tensor &x, const Tensor &weight) {
    if (x->ndim() != 2 || weight->ndim() != 2) {
        throw std::runtime_error("deepseek_v4_linear_bf16_fp32 expects 2D input and weight tensors.");
    }
    if (x->size(1) != weight->size(1)) {
        throw std::runtime_error("deepseek_v4_linear_bf16_fp32 input/weight K dimension mismatch.");
    }
    auto out = Tensor::empty({x->size(0), weight->size(0)}, DataType::F32, x->device());
    deepseek_v4_linear_bf16_fp32_(out, x, weight);
    return out;
}

void deepseek_v4_linear_bf16_fp32_(Tensor out, const Tensor &x, const Tensor &weight) {
    deepseek_v4_linear_bf16_fp32_kernel_(out, x, weight);
}

Tensor deepseek_v4_linear_bf16_fp32_kernel(const Tensor &x, const Tensor &weight) {
    if (x->ndim() != 2 || weight->ndim() != 2) {
        throw std::runtime_error("deepseek_v4_linear_bf16_fp32_kernel expects 2D input and weight tensors.");
    }
    if (x->size(1) != weight->size(1)) {
        throw std::runtime_error("deepseek_v4_linear_bf16_fp32_kernel input/weight K dimension mismatch.");
    }
    auto out = Tensor::empty({x->size(0), weight->size(0)}, DataType::F32, x->device());
    deepseek_v4_linear_bf16_fp32_kernel_(out, x, weight);
    return out;
}

} // namespace infinicore::op
