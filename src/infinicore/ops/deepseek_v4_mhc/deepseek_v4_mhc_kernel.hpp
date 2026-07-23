#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_mhc {

void launch_pre_kernel(void *y,
                       float *post,
                       float *comb,
                       const void *x,
                       const float *fn,
                       const float *scale,
                       const float *base,
                       float *mixes,
                       float *sqsum,
                       float *pre,
                       int64_t tokens,
                       int64_t hc,
                       int64_t hidden,
                       double rms_eps,
                       double hc_eps,
                       int sinkhorn_iters,
                       void *stream);

void launch_post_kernel(void *y,
                        const void *x,
                        const void *residual,
                        const float *post,
                        const float *comb,
                        int64_t tokens,
                        int64_t hc,
                        int64_t hidden,
                        void *stream);

void launch_head_kernel(void *y,
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

} // namespace infinicore::op::deepseek_v4_mhc
