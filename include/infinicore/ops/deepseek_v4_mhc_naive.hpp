#pragma once

#include "common/op.hpp"

namespace infinicore::op {

void deepseek_v4_mhc_pre_naive_(Tensor y,
                                Tensor post,
                                Tensor comb,
                                const Tensor &x,
                                const Tensor &fn,
                                const Tensor &scale,
                                const Tensor &base,
                                double rms_eps,
                                double hc_eps,
                                int sinkhorn_iters);

void deepseek_v4_mhc_post_naive_(Tensor y,
                                 const Tensor &x,
                                 const Tensor &residual,
                                 const Tensor &post,
                                 const Tensor &comb);

void deepseek_v4_mhc_head_naive_(Tensor y,
                                 const Tensor &x,
                                 const Tensor &fn,
                                 const Tensor &scale,
                                 const Tensor &base,
                                 double rms_eps,
                                 double hc_eps);

} // namespace infinicore::op
