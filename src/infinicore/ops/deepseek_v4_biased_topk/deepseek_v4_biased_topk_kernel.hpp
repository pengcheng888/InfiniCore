#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_biased_topk {

void launch_biased_topk_generic(float *topk_weights,
                                int32_t *topk_indices,
                                const float *router_logits,
                                const float *correction_bias,
                                int64_t tokens,
                                int64_t num_experts,
                                int64_t topk,
                                bool renormalize,
                                void *stream);

void launch_biased_topk_dsv4(float *topk_weights,
                             int32_t *topk_indices,
                             const float *router_logits,
                             const float *correction_bias,
                             int64_t tokens,
                             void *stream);

void launch_biased_topk(float *topk_weights,
                        int32_t *topk_indices,
                        const float *router_logits,
                        const float *correction_bias,
                        int64_t tokens,
                        int64_t num_experts,
                        int64_t topk,
                        bool renormalize,
                        void *stream);

} // namespace infinicore::op::deepseek_v4_biased_topk
