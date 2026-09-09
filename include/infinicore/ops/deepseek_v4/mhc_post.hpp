#pragma once

#include "../../device.hpp"
#include "../../graph/graph.hpp"
#include "../common/op.hpp"

namespace infinicore::op::deepseek_v4 {


INFINICORE_GRAPH_OP_CLASS(MhcPost,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &);


void mhc_post_(Tensor y,
                           const Tensor &x,
                           const Tensor &residual,
                           const Tensor &post,
                           const Tensor &comb);

void mhc_post_aten_(Tensor y,
                                const Tensor &x,
                                const Tensor &residual,
                                const Tensor &post,
                                const Tensor &comb);

void mhc_post_kernel_(Tensor y,
                                  const Tensor &x,
                                  const Tensor &residual,
                                  const Tensor &post,
                                  const Tensor &comb);

} // namespace infinicore::op
