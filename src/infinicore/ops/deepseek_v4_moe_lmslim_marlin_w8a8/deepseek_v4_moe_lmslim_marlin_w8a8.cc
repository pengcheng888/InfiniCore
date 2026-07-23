#include "infinicore/ops/deepseek_v4_moe_lmslim_marlin_w8a8.hpp"

#include "infinicore/ops/deepseek_v4_fused_experts_impl_int8_marlin.hpp"

namespace infinicore::op {

void deepseek_v4_moe_lmslim_marlin_w8a8_(Tensor output,
                                         const Tensor &hidden_states,
                                         const Tensor &w1,
                                         const Tensor &w2,
                                         const Tensor &topk_weights,
                                         const Tensor &topk_ids,
                                         const Tensor &w1_scale,
                                         const Tensor &w2_scale,
                                         int64_t global_num_experts,
                                         double routed_scaling_factor) {
    deepseek_v4_fused_experts_impl_int8_marlin_(
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        global_num_experts,
        routed_scaling_factor,
        false);
    return;
}

} // namespace infinicore::op
