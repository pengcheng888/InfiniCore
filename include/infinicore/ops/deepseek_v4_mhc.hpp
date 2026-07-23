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

void deepseek_v4_mhc_pre_kernel_(Tensor y,
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

void deepseek_v4_mhc_post_kernel_(Tensor y,
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

void deepseek_v4_mhc_head_kernel_(Tensor y,
                           const Tensor &x,
                           const Tensor &fn,
                           const Tensor &scale,
                           const Tensor &base,
                           double rms_eps,
                           double hc_eps);

void deepseek_v4_moe_w8a8_naive_(Tensor y,
                                     const Tensor &x,
                                     const Tensor &topk_weights,
                                     const Tensor &topk_indices,
                                     const Tensor &w13,
                                     const Tensor &w13_scale,
                                     const Tensor &w2,
                                     const Tensor &w2_scale,
                                     double swiglu_limit);

} // namespace infinicore::op
