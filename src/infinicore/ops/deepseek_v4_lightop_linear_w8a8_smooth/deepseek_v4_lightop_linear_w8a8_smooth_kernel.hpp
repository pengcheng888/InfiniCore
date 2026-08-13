#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_lightop_linear_w8a8_smooth_impl {

void launch_w8a8_smooth_gemm_bf16(void *output,
                                  const int8_t *q_input,
                                  const int8_t *weight,
                                  const float *input_scale,
                                  const float *weight_scale,
                                  const void *bias,
                                  int64_t m,
                                  int64_t n,
                                  int64_t k,
                                  void *stream);

} // namespace infinicore::op::deepseek_v4_lightop_linear_w8a8_smooth_impl
