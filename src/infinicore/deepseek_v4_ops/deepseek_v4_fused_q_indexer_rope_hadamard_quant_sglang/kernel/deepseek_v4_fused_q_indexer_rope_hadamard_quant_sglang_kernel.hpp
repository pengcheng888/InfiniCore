#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang {

void launch_fused_q_indexer_rope_hadamard_quant_sglang(const void *q_input,
                                                       uint8_t *q_fp8,
                                                       const void *weight,
                                                       float *weights_out,
                                                       float weight_scale,
                                                       const float *freqs_cis,
                                                       const void *positions,
                                                       bool positions_i64,
                                                       int64_t rows,
                                                       int64_t heads,
                                                       void *stream);

} // namespace infinicore::op::deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang
