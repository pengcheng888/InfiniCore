#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_hash_topk {

void launch_hash_topk_generic(float *topk_weights,
                              int32_t *topk_indices,
                              const float *router_logits,
                              const int64_t *input_ids,
                              const void *tid2eid,
                              bool tid2eid_i64,
                              int64_t tokens,
                              int64_t num_experts,
                              int64_t topk,
                              int64_t num_fused_shared_experts,
                              float routed_scaling_factor,
                              void *stream);

void launch_hash_topk_sglang(float *topk_weights,
                             int32_t *topk_indices,
                             const float *router_logits,
                             const int64_t *input_ids,
                             const void *tid2eid,
                             bool tid2eid_i64,
                             int64_t tokens,
                             int64_t num_experts,
                             int64_t topk,
                             int64_t num_fused_shared_experts,
                             float routed_scaling_factor,
                             void *stream);

void launch_hash_topk_num_experts_256_topk_6_(float *topk_weights,
                                              int32_t *topk_indices,
                                              const float *router_logits,
                                              const int64_t *input_ids,
                                              const void *tid2eid,
                                              bool tid2eid_i64,
                                              int64_t tokens,
                                              int64_t num_fused_shared_experts,
                                              float routed_scaling_factor,
                                              void *stream);

void launch_hash_topk(float *topk_weights,
                      int32_t *topk_indices,
                      const float *router_logits,
                      const int64_t *input_ids,
                      const void *tid2eid,
                      bool tid2eid_i64,
                      int64_t tokens,
                      int64_t num_experts,
                      int64_t topk,
                      int64_t num_fused_shared_experts,
                      float routed_scaling_factor,
                      void *stream);

} // namespace infinicore::op::deepseek_v4_hash_topk
