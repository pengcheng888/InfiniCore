#include "infinicore/ops/qwen3_add_rms_norm.hpp"

#include "infinicore/ops/add_rms_norm.hpp"

namespace infinicore::op {

std::pair<Tensor, Tensor> qwen3_add_rms_norm(const Tensor &a, const Tensor &b, const Tensor &weight, float epsilon) {
    return add_rms_norm(a, b, weight, epsilon);
}

void qwen3_add_rms_norm_(Tensor out, Tensor residual, const Tensor &a, const Tensor &b, const Tensor &weight, float epsilon) {
    add_rms_norm_(out, residual, a, b, weight, epsilon);
}

void qwen3_add_rms_norm_inplace(Tensor input, Tensor residual, const Tensor &weight, float epsilon) {
    add_rms_norm_inplace(input, residual, weight, epsilon);
}

} // namespace infinicore::op
