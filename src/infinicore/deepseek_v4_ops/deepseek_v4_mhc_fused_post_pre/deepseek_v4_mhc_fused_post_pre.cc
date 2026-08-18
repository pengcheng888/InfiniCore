#include "infinicore/ops/deepseek_v4_mhc_fused_post_pre.hpp"

namespace infinicore::op {

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
                                     double norm_eps) {
    deepseek_v4_mhc_fused_post_pre_kernel_(residual_cur,
                                           post_mix_cur,
                                           comb_mix_cur,
                                           layer_input_cur,
                                           x,
                                           residual,
                                           post_layer_mix,
                                           comb_res_mix,
                                           fn,
                                           hc_scale,
                                           hc_base,
                                           rms_eps,
                                           hc_pre_eps,
                                           hc_sinkhorn_eps,
                                           hc_post_mult_value,
                                           sinkhorn_repeat,
                                           norm_weight,
                                           norm_eps);
}

} // namespace infinicore::op
