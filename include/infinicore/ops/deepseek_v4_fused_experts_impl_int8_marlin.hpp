#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

#include <optional>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4FusedExpertsImplInt8Marlin,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          int64_t,
                          double,
                          bool,
                          std::optional<Tensor>);

void deepseek_v4_fused_experts_impl_int8_marlin_(Tensor output,
                                                 const Tensor &hidden_states,
                                                 const Tensor &w1,
                                                 const Tensor &w2,
                                                 const Tensor &topk_weights,
                                                 const Tensor &topk_ids,
                                                 const Tensor &w1_scale,
                                                 const Tensor &w2_scale,
                                                 int64_t global_num_experts,
                                                 double routed_scaling_factor = 1.0,
                                                 bool inplace = false,
                                                 const std::optional<Tensor> &shared_output = std::nullopt);

void deepseek_v4_python_fused_experts_impl_int8_marlin_(Tensor output,
                                                        const Tensor &hidden_states,
                                                        const Tensor &w1,
                                                        const Tensor &w2,
                                                        const Tensor &topk_weights,
                                                        const Tensor &topk_ids,
                                                        const Tensor &w1_scale,
                                                        const Tensor &w2_scale,
                                                        int64_t global_num_experts,
                                                        double routed_scaling_factor = 1.0,
                                                        bool inplace = false);

} // namespace infinicore::op
