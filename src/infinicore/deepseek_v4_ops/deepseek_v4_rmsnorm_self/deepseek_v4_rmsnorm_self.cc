#include "infinicore/ops/deepseek_v4_rmsnorm_self.hpp"

namespace infinicore::op {

Tensor deepseek_v4_rmsnorm_self(const Tensor &x, float epsilon) {
    auto out = Tensor::empty(x->shape(), x->dtype(), x->device());
    deepseek_v4_rmsnorm_self_kernel_(out, x, epsilon);
    return out;
}

void deepseek_v4_rmsnorm_self_(Tensor out, const Tensor &x, float epsilon) {
    deepseek_v4_rmsnorm_self_kernel_(out, x, epsilon);
}

} // namespace infinicore::op
