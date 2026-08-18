#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

namespace deepseek_v4 {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4MhcFusedPostPre,
                          Tensor,
                          Tensor,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          double,
                          double,
                          double,
                          double,
                          int,
                          const Tensor &,
                          double);

} // namespace deepseek_v4

void deepseek_v4_mhc_fused_post_pre_(Tensor residual_cur,
                                     Tensor post_mix_cur,
                                     Tensor comb_mix_cur,
                                     Tensor layer_input_cur,
                                     const Tensor &x,
                                     const Tensor &residual,
                                     const Tensor &post_layer_mix,
                                     const Tensor &comb_res_mix,
                                     const Tensor &fn,
                                     const Tensor &hc_scale,
                                     const Tensor &hc_base,
                                     double rms_eps,
                                     double hc_pre_eps,
                                     double hc_sinkhorn_eps,
                                     double hc_post_mult_value,
                                     int sinkhorn_repeat,
                                     const Tensor &norm_weight,
                                     double norm_eps);

void deepseek_v4_mhc_fused_post_pre_aten_(Tensor residual_cur,
                                          Tensor post_mix_cur,
                                          Tensor comb_mix_cur,
                                          Tensor layer_input_cur,
                                          const Tensor &x,
                                          const Tensor &residual,
                                          const Tensor &post_layer_mix,
                                          const Tensor &comb_res_mix,
                                          const Tensor &fn,
                                          const Tensor &hc_scale,
                                          const Tensor &hc_base,
                                          double rms_eps,
                                          double hc_pre_eps,
                                          double hc_sinkhorn_eps,
                                          double hc_post_mult_value,
                                          int sinkhorn_repeat,
                                          const Tensor &norm_weight,
                                          double norm_eps);

void deepseek_v4_mhc_fused_post_pre_kernel_(Tensor residual_cur,
                                            Tensor post_mix_cur,
                                            Tensor comb_mix_cur,
                                            Tensor layer_input_cur,
                                            const Tensor &x,
                                            const Tensor &residual,
                                            const Tensor &post_layer_mix,
                                            const Tensor &comb_res_mix,
                                            const Tensor &fn,
                                            const Tensor &hc_scale,
                                            const Tensor &hc_base,
                                            double rms_eps,
                                            double hc_pre_eps,
                                            double hc_sinkhorn_eps,
                                            double hc_post_mult_value,
                                            int sinkhorn_repeat,
                                            const Tensor &norm_weight,
                                            double norm_eps);

} // namespace infinicore::op
