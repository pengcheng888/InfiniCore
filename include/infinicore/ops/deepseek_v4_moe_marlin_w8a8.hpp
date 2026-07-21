#pragma once

#include "common/op.hpp"

#include <optional>

namespace infinicore::op {

void deepseek_v4_moe_marlin_w8a8_(const Tensor &input,
                                  const Tensor &b_qweight,
                                  Tensor output,
                                  const Tensor &a_scale,
                                  const Tensor &b_scale,
                                  const std::optional<Tensor> &topk_weights,
                                  const Tensor &sorted_token_ids,
                                  const Tensor &expert_ids,
                                  const Tensor &num_tokens_post_pad,
                                  int top_k,
                                  int mode,
                                  int delta);

void deepseek_v4_moe_marlin_w8a8_fp8_(const Tensor &input,
                                      const Tensor &b_qweight,
                                      Tensor output,
                                      const Tensor &a_scale,
                                      const Tensor &b_scale,
                                      const std::optional<Tensor> &topk_weights,
                                      const Tensor &sorted_token_ids,
                                      const Tensor &expert_ids,
                                      const Tensor &num_tokens_post_pad,
                                      int top_k,
                                      int mode,
                                      int delta);

} // namespace infinicore::op
