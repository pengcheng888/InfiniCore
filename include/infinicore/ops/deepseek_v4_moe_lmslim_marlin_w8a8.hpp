#pragma once

#include "common/op.hpp"

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
                                          double routed_scaling_factor = 1.0);

} // namespace infinicore::op
