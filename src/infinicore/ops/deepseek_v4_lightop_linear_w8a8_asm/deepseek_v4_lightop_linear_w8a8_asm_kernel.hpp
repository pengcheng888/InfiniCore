#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_lightop_linear_w8a8_asm_impl {

void launch_prepare_per_channel_scales(float *input_block_scale,
                                       float *weight_block_scale,
                                       const float *input_scale,
                                       int64_t m,
                                       int64_t n,
                                       int64_t k,
                                       void *stream);

void launch_per_token_quant_int8_bf16(int8_t *q_input,
                                      float *input_scale,
                                      const void *input,
                                      const float *smooth_scale,
                                      int64_t m,
                                      int64_t k,
                                      void *stream);

void launch_apply_per_channel_weight_scale(void *output,
                                           const float *weight_scale,
                                           int64_t m,
                                           int64_t n,
                                           void *stream);

} // namespace infinicore::op::deepseek_v4_lightop_linear_w8a8_asm_impl
