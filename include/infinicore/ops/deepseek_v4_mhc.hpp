#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"
#include "deepseek_v4_mhc_naive.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4MhcPreKernel,
                          Tensor,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          double,
                          double,
                          int);

INFINICORE_GRAPH_OP_CLASS(DeepseekV4MhcPostKernel,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &);

INFINICORE_GRAPH_OP_CLASS(DeepseekV4MhcHeadKernel,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          double,
                          double);

void deepseek_v4_mhc_pre_(Tensor y,
                       Tensor post,
                       Tensor comb,
                       const Tensor &x,
                       const Tensor &fn,
                       const Tensor &scale,
                       const Tensor &base,
                       double rms_eps,
                       double hc_eps,
                       int sinkhorn_iters);

void deepseek_v4_mhc_post_(Tensor y,
                           const Tensor &x,
                           const Tensor &residual,
                           const Tensor &post,
                           const Tensor &comb);

void deepseek_v4_mhc_head_(Tensor y,
                           const Tensor &x,
                           const Tensor &fn,
                           const Tensor &scale,
                           const Tensor &base,
                           double rms_eps,
                           double hc_eps);

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

void deepseek_v4_mhc_post_kernel_(Tensor y,
                           const Tensor &x,
                           const Tensor &residual,
                           const Tensor &post,
                           const Tensor &comb);

void deepseek_v4_mhc_head_kernel_(Tensor y,
                           const Tensor &x,
                           const Tensor &fn,
                           const Tensor &scale,
                           const Tensor &base,
                           double rms_eps,
                           double hc_eps);

} // namespace infinicore::op
