#pragma once

#include "common/op.hpp"

namespace infinicore::op {

void deepseek_v4_hash_topk_naive_(Tensor topk_weights,
                            Tensor topk_indices,
                            const Tensor &router_logits,
                            const Tensor &input_ids,
                            const Tensor &tid2eid,
                            bool renormalize);

void deepseek_v4_hash_topk_kernel_(Tensor topk_weights,
                                   Tensor topk_indices,
                                   const Tensor &router_logits,
                                   const Tensor &input_ids,
                                   const Tensor &tid2eid,
                                   bool renormalize);

void deepseek_v4_hash_topk_generic_kernel_(Tensor topk_weights,
                                           Tensor topk_indices,
                                           const Tensor &router_logits,
                                           const Tensor &input_ids,
                                           const Tensor &tid2eid,
                                           bool renormalize);

} // namespace infinicore::op
