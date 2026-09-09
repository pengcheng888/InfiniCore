#include "infinicore/ops/deepseek_v4/mhc_pre.hpp"

namespace infinicore::op::deepseek_v4 {

void mhc_pre_(Tensor y,
              Tensor post,
              Tensor comb,
              const Tensor &residual,
              const Tensor &fn,
              const Tensor &hc_scale,
              const Tensor &hc_base,
              double rms_eps,
              double hc_pre_eps,
              double hc_sinkhorn_eps,
              int sinkhorn_repeat) {
    mhc_pre_kernel_(y, post, comb, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, sinkhorn_repeat);
    return;
}

} // namespace infinicore::op::deepseek_v4
