#include "infinicore/ops/deepseek_v4_hc_head.hpp"

namespace infinicore::op {

void deepseek_v4_hc_head_(Tensor y,
                          const Tensor &x,
                          const Tensor &fn,
                          const Tensor &scale,
                          const Tensor &base,
                          double rms_eps,
                          double hc_eps) {
    deepseek_v4_hc_head_kernel_(y, x, fn, scale, base, rms_eps, hc_eps);
    return;
}

} // namespace infinicore::op
