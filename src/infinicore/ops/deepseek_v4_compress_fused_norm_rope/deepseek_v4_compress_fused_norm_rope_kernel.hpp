#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_compress_fused_norm_rope_kernel {

void launch_compress_fused_norm_rope(void *input,
                                     int input_dtype,
                                     const void *norm_weight,
                                     int norm_weight_dtype,
                                     const float *freqs_cis,
                                     const void *positions,
                                     bool positions_i64,
                                     int64_t tokens,
                                     int64_t dim,
                                     float epsilon,
                                     void *stream);

} // namespace infinicore::op::deepseek_v4_compress_fused_norm_rope_kernel
