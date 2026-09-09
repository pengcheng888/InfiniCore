#pragma once

#include "../common/op.hpp"

#include <cstdint>
#include <string>

namespace infinicore::op::deepseek_v4 {

void hash_topk_(Tensor topk_weights,
                Tensor topk_indices,
                const Tensor &router_logits,
                const Tensor &input_ids,
                const Tensor &tid2eid,
                int64_t num_fused_shared_experts,
                float routed_scaling_factor,
                const std::string &scoring_func);

void hash_topk_aten_(Tensor topk_weights,
                     Tensor topk_indices,
                     const Tensor &router_logits,
                     const Tensor &input_ids,
                     const Tensor &tid2eid,
                     int64_t num_fused_shared_experts,
                     float routed_scaling_factor,
                     const std::string &scoring_func);

} // namespace infinicore::op::deepseek_v4
