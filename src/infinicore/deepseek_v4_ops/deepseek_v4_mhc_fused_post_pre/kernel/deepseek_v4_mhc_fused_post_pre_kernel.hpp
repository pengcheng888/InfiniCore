#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_mhc_fused_post_pre {

void launch_kernel(void *residual_cur,
                   float *post_mix_cur,
                   float *comb_mix_cur,
                   void *layer_input_cur,
                   const void *x,
                   const void *residual,
                   const float *post_layer_mix,
                   const float *comb_res_mix,
                   const float *fn,
                   const float *hc_scale,
                   const float *hc_base,
                   float *mixes,
                   float *sqsum,
                   float *pre,
                   float *mixes_partial,
                   float *sqsum_partial,
                   int64_t tokens,
                   int64_t hc,
                   int64_t hidden,
                   double rms_eps,
                   double hc_pre_eps,
                   double hc_sinkhorn_eps,
                   double hc_post_mult_value,
                   int sinkhorn_repeat,
                   const void *norm_weight,
                   double norm_eps,
                   void *stream);

} // namespace infinicore::op::deepseek_v4_mhc_fused_post_pre
