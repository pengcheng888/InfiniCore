#pragma once

#include "common/op.hpp"

#include <optional>

namespace infinicore::op {

void deepseek_v4_lightop_moe_align_block_size_(const Tensor &topk_ids,
                                                int num_experts,
                                                int block_size,
                                                Tensor sorted_token_ids,
                                                Tensor expert_ids,
                                                Tensor num_tokens_post_pad,
                                                bool is_fuse_fill);

void deepseek_v4_lightop_moe_gemm_marlin_w8a8_(const Tensor &input,
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

void deepseek_v4_lightop_fuse_silu_mul_quant_(Tensor output,
                                               Tensor scales,
                                               const Tensor &input,
                                               const std::optional<Tensor> &num_local_tokens_tensor,
                                               int topk,
                                               int expect_m,
                                               const std::optional<Tensor> &expert_ids);

void deepseek_v4_lightop_moe_sum_(Tensor output,
                                   const Tensor &input,
                                   const std::optional<Tensor> &bias,
                                   const std::optional<Tensor> &expert_mask,
                                   const std::optional<Tensor> &num_local_tokens,
                                   float factor,
                                   int expect_m);

} // namespace infinicore::op
