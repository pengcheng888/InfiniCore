#include "infinicore/ops/deepseek_v4_mhc_fused_post_pre.hpp"

#include "infinicore/ops/deepseek_v4_mhc_post.hpp"
#include "infinicore/ops/deepseek_v4_mhc_pre.hpp"
#include "infinicore/ops/deepseek_v4_rms_norm.hpp"

#include <stdexcept>

namespace infinicore::op {

void deepseek_v4_mhc_fused_post_pre_naive_(Tensor residual_cur,
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
    if (hc_post_mult_value != 2.0) {
        throw std::runtime_error("deepseek_v4_mhc_fused_post_pre_naive_ currently expects hc_post_mult_value == 2.0.");
    }
    deepseek_v4_mhc_post_naive_(residual_cur, x, residual, post_layer_mix, comb_res_mix);
    deepseek_v4_mhc_pre_naive_(layer_input_cur,
                               post_mix_cur,
                               comb_mix_cur,
                               residual_cur,
                               fn,
                               hc_scale,
                               hc_base,
                               rms_eps,
                               hc_pre_eps,
                               hc_sinkhorn_eps,
                               sinkhorn_repeat);
    deepseek_v4_rms_norm_(layer_input_cur, layer_input_cur, norm_weight, static_cast<float>(norm_eps));
}

} // namespace infinicore::op
