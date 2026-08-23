#include "infinicore/ops/qwen3_rms_norm.hpp"

#include "infinicore/ops/rms_norm.hpp"

namespace infinicore::op {

Tensor qwen3_rms_norm(const Tensor &x, const Tensor &weight, float epsilon) {
    return rms_norm(x, weight, epsilon);
}

void qwen3_rms_norm_(Tensor y, const Tensor &x, const Tensor &weight, float epsilon) {
    rms_norm_(y, x, weight, epsilon);
}

} // namespace infinicore::op
