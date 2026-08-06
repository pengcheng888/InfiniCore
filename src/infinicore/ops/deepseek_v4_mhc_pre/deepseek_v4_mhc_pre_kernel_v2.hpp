#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_mhc_pre_v2 {

void launch_kernel(void *y,
                   float *post,
                   float *comb,
                   const void *residual,
                   const float *fn,
                   const float *hc_scale,
                   const float *hc_base,
                   float *partial_mixes,
                   float *partial_sqsum,
                   int64_t tokens,
                   int64_t hc,
                   int64_t hidden,
                   double rms_eps,
                   double hc_pre_eps,
                   double hc_sinkhorn_eps,
                   int sinkhorn_repeat,
                   int split_k,
                   int partial_stride,
                   void *stream);

} // namespace infinicore::op::deepseek_v4_mhc_pre_v2
