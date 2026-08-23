#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_fused_q_norm_rope {

void launch_fused_q_norm_rope(void *q_out,
                              const void *q_input,
                              const float *freqs_cis,
                              const void *positions,
                              bool positions_i64,
                              int64_t tokens,
                              int64_t heads,
                              int64_t q_input_stride_batch,
                              int64_t q_out_stride_batch,
                              float epsilon,
                              void *stream);

} // namespace infinicore::op::deepseek_v4_fused_q_norm_rope
