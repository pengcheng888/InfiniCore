#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_fused_experts_impl_int8_marlin {

void launch_per_token_quant_int8_bf16(void *output,
                                      float *scale,
                                      const void *input,
                                      int64_t rows,
                                      int64_t cols,
                                      void *stream);

void launch_moe_sum_scale_add_bf16(void *output,
                                   const void *input,
                                   const void *shared_output,
                                   int64_t tokens,
                                   int64_t topk,
                                   int64_t hidden,
                                   float factor,
                                   void *stream);

} // namespace infinicore::op::deepseek_v4_fused_experts_impl_int8_marlin
