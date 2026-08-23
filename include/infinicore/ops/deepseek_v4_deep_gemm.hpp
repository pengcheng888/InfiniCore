#pragma once

#include "common/op.hpp"

#include <optional>

namespace infinicore::op {

void deepseek_v4_deep_gemm_low_latency_grouped_gemm_(const Tensor &matrix_a,
                                                     const Tensor &matrix_b,
                                                     const Tensor &matrix_a_scale,
                                                     const Tensor &matrix_b_scale,
                                                     const Tensor &actual_tokens,
                                                     Tensor matrix_c,
                                                     int max_tokens,
                                                     int experts,
                                                     int cu_s,
                                                     bool block_wise,
                                                     bool b_overlap,
                                                     const std::optional<Tensor> &signal);

void deepseek_v4_deep_gemm_moe_w8a8_i8_marlin_prefill_down_(const Tensor &input,
                                                            const Tensor &b_qweight,
                                                            Tensor output,
                                                            const Tensor &a_scale,
                                                            const Tensor &b_scale,
                                                            const Tensor &topk_weights,
                                                            const Tensor &sorted_token_ids,
                                                            const Tensor &expert_ids,
                                                            const Tensor &num_tokens_post_pad,
                                                            int top_k,
                                                            int real_topk);

void deepseek_v4_deep_gemm_moe_w8a8_marlin_decode_down_fp8_(const Tensor &input,
                                                            const Tensor &b_qweight,
                                                            Tensor output,
                                                            const Tensor &a_scale,
                                                            const Tensor &b_scale,
                                                            const Tensor &topk_weights,
                                                            const Tensor &sorted_token_ids,
                                                            const Tensor &expert_ids,
                                                            const Tensor &num_tokens_post_pad,
                                                            int top_k,
                                                            int real_topk);

} // namespace infinicore::op
