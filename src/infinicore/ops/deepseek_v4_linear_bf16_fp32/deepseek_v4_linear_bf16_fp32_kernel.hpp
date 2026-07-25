#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_linear_bf16_fp32_impl {

void launch_linear_bf16_fp32(float *out,
                             const void *x,
                             const void *weight,
                             int64_t tokens,
                             int64_t out_features,
                             int64_t in_features,
                             void *stream);

} // namespace infinicore::op::deepseek_v4_linear_bf16_fp32_impl
