#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_mhc_pre {

void launch_kernel(void *y,
                   float *post,
                   float *comb,
                   const void *residual,
                   const float *fn,
                   const float *hc_scale,
                   const float *hc_base,
                   float *mixes,
                   float *sqsum,
                   float *pre,
                   int64_t tokens,
                   int64_t hc,
                   int64_t hidden,
                   double rms_eps,
                   double hc_pre_eps,
                   double hc_sinkhorn_eps,
                   int sinkhorn_repeat,
                   void *stream);

} // namespace infinicore::op::deepseek_v4_mhc_pre
