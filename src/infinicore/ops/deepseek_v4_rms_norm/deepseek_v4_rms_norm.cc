#include "infinicore/ops/deepseek_v4_rms_norm.hpp"

#include "infinicore/ops/rms_norm.hpp"

namespace infinicore::op {

Tensor deepseek_v4_rms_norm(const Tensor &x, const Tensor &weight, float epsilon) {
    return rms_norm(x, weight, epsilon);
}

void deepseek_v4_rms_norm_(Tensor y, const Tensor &x, const Tensor &weight, float epsilon) {
    rms_norm_(y, x, weight, epsilon);
}

} // namespace infinicore::op
