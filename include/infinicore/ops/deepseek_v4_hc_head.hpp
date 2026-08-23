#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

namespace deepseek_v4 {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4HcHead,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          double,
                          double);

} // namespace deepseek_v4

void deepseek_v4_hc_head_(Tensor y,
                           const Tensor &x,
                           const Tensor &fn,
                           const Tensor &scale,
                           const Tensor &base,
                           double rms_eps,
                           double hc_eps);

void deepseek_v4_hc_head_aten_(Tensor y,
                               const Tensor &x,
                               const Tensor &fn,
                               const Tensor &scale,
                               const Tensor &base,
                               double rms_eps,
                               double hc_eps);

void deepseek_v4_hc_head_naive_(Tensor y,
                                 const Tensor &x,
                                 const Tensor &fn,
                                 const Tensor &scale,
                                 const Tensor &base,
                                 double rms_eps,
                                 double hc_eps);

void deepseek_v4_hc_head_kernel_(Tensor y,
                                  const Tensor &x,
                                  const Tensor &fn,
                                  const Tensor &scale,
                                  const Tensor &base,
                                  double rms_eps,
                                  double hc_eps);

} // namespace infinicore::op
