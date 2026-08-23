#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_hc_head {

void launch_kernel(void *y,
                   const void *x,
                   const float *fn,
                   const float *scale,
                   const float *base,
                   float *mixes,
                   float *sqsum,
                   int64_t tokens,
                   int64_t hc,
                   int64_t hidden,
                   double rms_eps,
                   double hc_eps,
                   void *stream);

} // namespace infinicore::op::deepseek_v4_hc_head
