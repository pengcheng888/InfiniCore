#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

#include <cstddef>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4SharedExpertsImplInt8Marlin,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          int,
                          int,
                          int);

INFINICORE_GRAPH_OP_CLASS(DeepseekV4SharedExpertsImplInt8MarlinWorkspace,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          Tensor,
                          Tensor,
                          Tensor,
                          Tensor,
                          Tensor,
                          Tensor,
                          Tensor,
                          Tensor,
                          Tensor,
                          int,
                          int,
                          int);

void deepseek_v4_shared_experts_impl_int8_marlin_(Tensor output,
                                                  const Tensor &hidden_states,
                                                  const Tensor &w1,
                                                  const Tensor &w2,
                                                  const Tensor &w1_scale,
                                                  const Tensor &w2_scale,
                                                  int gemm1_mode = -1,
                                                  int gemm2_mode = -1,
                                                  int delta = 1);

void deepseek_v4_shared_experts_impl_int8_marlin_prepare_metadata_(Tensor sorted_token_ids,
                                                                   Tensor expert_ids,
                                                                   Tensor num_tokens_post_pad,
                                                                   Tensor topk_weights,
                                                                   size_t tokens);

void deepseek_v4_shared_experts_impl_int8_marlin_(Tensor output,
                                                  const Tensor &hidden_states,
                                                  const Tensor &w1,
                                                  const Tensor &w2,
                                                  const Tensor &w1_scale,
                                                  const Tensor &w2_scale,
                                                  Tensor sorted_token_ids,
                                                  Tensor expert_ids,
                                                  Tensor num_tokens_post_pad,
                                                  Tensor topk_weights,
                                                  Tensor q_hidden,
                                                  Tensor hidden_scale,
                                                  Tensor gate_up,
                                                  Tensor q_activated,
                                                  Tensor activated_scale,
                                                  int gemm1_mode = -1,
                                                  int gemm2_mode = -1,
                                                  int delta = 1);

} // namespace infinicore::op
