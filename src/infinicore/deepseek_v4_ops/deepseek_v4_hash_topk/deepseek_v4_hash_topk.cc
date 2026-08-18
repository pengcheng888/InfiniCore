#include "infinicore/ops/deepseek_v4_hash_topk.hpp"

namespace infinicore::op {

void deepseek_v4_hash_topk_(Tensor topk_weights,
                            Tensor topk_indices,
                            const Tensor &router_logits,
                            const Tensor &input_ids,
                            const Tensor &tid2eid,
                            int64_t num_fused_shared_experts,
                            float routed_scaling_factor,
                            const std::string &scoring_func) {
    deepseek_v4_hash_topk_kernel_(topk_weights,
                                  topk_indices,
                                  router_logits,
                                  input_ids,
                                  tid2eid,
                                  num_fused_shared_experts,
                                  routed_scaling_factor,
                                  scoring_func);
}

} // namespace infinicore::op
