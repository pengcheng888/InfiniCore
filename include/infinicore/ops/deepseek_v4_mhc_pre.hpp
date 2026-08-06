#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

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
                          double,
                          int);

INFINICORE_GRAPH_OP_CLASS(DeepseekV4MhcPreKernelV2,
                          Tensor,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          double,
                          double,
                          double,
                          int);

void deepseek_v4_mhc_pre_(Tensor y,
                          Tensor post,
                          Tensor comb,
                          const Tensor &residual,
                          const Tensor &fn,
                          const Tensor &hc_scale,
                          const Tensor &hc_base,
                          double rms_eps,
                          double hc_pre_eps,
                          double hc_sinkhorn_eps,
                          int sinkhorn_repeat);

void deepseek_v4_mhc_pre_naive_(Tensor y,
                                Tensor post,
                                Tensor comb,
                                const Tensor &residual,
                                const Tensor &fn,
                                const Tensor &hc_scale,
                                const Tensor &hc_base,
                                double rms_eps,
                                double hc_pre_eps,
                                double hc_sinkhorn_eps,
                                int sinkhorn_repeat);

void deepseek_v4_mhc_pre_kernel_(Tensor y,
                                 Tensor post,
                                 Tensor comb,
                                 const Tensor &residual,
                                 const Tensor &fn,
                                 const Tensor &hc_scale,
                                 const Tensor &hc_base,
                                 double rms_eps,
                                 double hc_pre_eps,
                                 double hc_sinkhorn_eps,
                                 int sinkhorn_repeat);

void deepseek_v4_mhc_pre_kernel_v2_(Tensor y,
                                    Tensor post,
                                    Tensor comb,
                                    const Tensor &residual,
                                    const Tensor &fn,
                                    const Tensor &hc_scale,
                                    const Tensor &hc_base,
                                    double rms_eps,
                                    double hc_pre_eps,
                                    double hc_sinkhorn_eps,
                                    int sinkhorn_repeat);

} // namespace infinicore::op
