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
                              bool renormalize,
                              void *stream);

void launch_hash_topk_dsv4(float *topk_weights,
                           int32_t *topk_indices,
                           const float *router_logits,
                           const int64_t *input_ids,
                           const void *tid2eid,
                           bool tid2eid_i64,
                           int64_t tokens,
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
                      bool renormalize,
                      void *stream);

} // namespace infinicore::op::deepseek_v4_hash_topk
