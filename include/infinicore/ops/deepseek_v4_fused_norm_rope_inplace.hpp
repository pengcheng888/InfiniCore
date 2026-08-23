#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4FusedNormRopeInplaceKernel,
                          Tensor,
                          const Tensor &,
                          float,
                          const Tensor &,
                          const Tensor &);

void deepseek_v4_fused_norm_rope_inplace_naive_(Tensor input,
                                                const Tensor &norm_weight,
                                                float epsilon,
                                                const Tensor &freqs_cis,
                                                const Tensor &positions);

void deepseek_v4_fused_norm_rope_inplace_kernel_(Tensor input,
                                                 const Tensor &norm_weight,
                                                 float epsilon,
                                                 const Tensor &freqs_cis,
                                                 const Tensor &positions);

void deepseek_v4_fused_norm_rope_inplace(Tensor input,
                                         const Tensor &norm_weight,
                                         float epsilon,
                                         const Tensor &freqs_cis,
                                         const Tensor &positions);

void deepseek_v4_fused_norm_rope_inplace_(Tensor input,
                                          const Tensor &norm_weight,
                                          float epsilon,
                                          const Tensor &freqs_cis,
                                          const Tensor &positions);

} // namespace infinicore::op
