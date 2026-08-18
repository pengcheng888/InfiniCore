#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_mhc_post {

void launch_kernel(void *y,
                   const void *x,
                   const void *residual,
                   const float *post,
                   const float *comb,
                   int64_t tokens,
                   int64_t hc,
                   int64_t hidden,
                   void *stream);

} // namespace infinicore::op::deepseek_v4_mhc_post
