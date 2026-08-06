#include "infinicore/ops/deepseek_v4_linear_bf16_fp32.hpp"

#include "deepseek_v4_linear_bf16_fp32_common.hpp"

#include "infinicore/dtype.hpp"

#include <stdexcept>
#include <string>

namespace infinicore::op {

namespace deepseek_v4_linear_bf16_fp32_impl {

void check_shapes(const Tensor &out, const Tensor &x, const Tensor &weight, const char *op_name) {
    if (x->ndim() != 2 || weight->ndim() != 2) {
        throw std::runtime_error(std::string(op_name) + " expects 2D input and weight tensors.");
    }
    if (x->size(1) != weight->size(1)) {
        throw std::runtime_error(std::string(op_name) + " input/weight K dimension mismatch.");
    }
    if (out->shape() != Shape{x->size(0), weight->size(0)}) {
        throw std::runtime_error(std::string(op_name) + " output shape mismatch.");
    }
    if (out->dtype() != DataType::F32) {
        throw std::runtime_error(std::string(op_name) + " output dtype must be float32.");
    }
}

} // namespace deepseek_v4_linear_bf16_fp32_impl

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
