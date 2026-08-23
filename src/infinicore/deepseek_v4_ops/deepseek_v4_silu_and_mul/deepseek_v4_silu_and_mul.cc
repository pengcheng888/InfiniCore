#include "infinicore/ops/deepseek_v4_silu_and_mul.hpp"

#include <stdexcept>

namespace infinicore::op {

Tensor deepseek_v4_silu_and_mul(const Tensor &x) {
    Shape shape = x->shape();
    if (shape.empty() || shape.back() % 2 != 0) {
        throw std::runtime_error("deepseek_v4_silu_and_mul input last dim must be even.");
    }
    shape.back() /= 2;
    auto out = Tensor::empty(shape, x->dtype(), x->device());
    deepseek_v4_silu_and_mul_(out, x);
    return out;
}

void deepseek_v4_silu_and_mul_(Tensor out, const Tensor &x) {
    deepseek_v4_silu_and_mul_kernel_(out, x);
}

} // namespace infinicore::op
