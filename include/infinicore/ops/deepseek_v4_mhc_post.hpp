#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4MhcPostKernel,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &);

void deepseek_v4_mhc_post_(Tensor y,
                           const Tensor &x,
                           const Tensor &residual,
                           const Tensor &post,
                           const Tensor &comb);

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

} // namespace infinicore::op
