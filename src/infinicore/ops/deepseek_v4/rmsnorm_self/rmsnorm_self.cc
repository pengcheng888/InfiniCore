#include "infinicore/ops/deepseek_v4/rmsnorm_self.hpp"

#include <stdexcept>

namespace infinicore::op::deepseek_v4 {

Tensor rmsnorm_self(const Tensor &x, float epsilon) {
    if (x->ndim() < 1) {
        throw std::runtime_error("deepseek_v4::rmsnorm_self expects rank >= 1.");
    }
    auto out = Tensor::empty(x->shape(), x->dtype(), x->device());
    rmsnorm_self_(out, x, epsilon);
    return out;
}

void rmsnorm_self_(Tensor out, const Tensor &x, float epsilon) {
    rmsnorm_self_kernel_(out, x, epsilon);
}

Tensor rmsnorm_self_aten(const Tensor &x, float epsilon) {
    if (x->ndim() < 1) {
        throw std::runtime_error("deepseek_v4::rmsnorm_self_aten expects rank >= 1.");
    }
    auto out = Tensor::empty(x->shape(), x->dtype(), x->device());
    rmsnorm_self_aten_(out, x, epsilon);
    return out;
}

} // namespace infinicore::op::deepseek_v4
