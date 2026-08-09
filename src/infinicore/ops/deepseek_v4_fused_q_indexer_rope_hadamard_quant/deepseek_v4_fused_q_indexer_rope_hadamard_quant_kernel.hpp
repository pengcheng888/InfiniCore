#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_fused_q_indexer_rope_hadamard_quant {

enum Dsv4ScalarType : int {
    kDsv4BF16 = 0,
    kDsv4F16 = 1,
    kDsv4F32 = 2,
};

void launch_fused_q_indexer_rope_hadamard_quant(const void *q,
                                                int q_dtype,
                                                const void *weights,
                                                int weights_dtype,
                                                uint8_t *q_fp8,
                                                float *q_scale,
                                                float *fused_weights,
                                                float weight_scale,
                                                const float *freqs_cis,
                                                const void *positions,
                                                bool positions_i64,
                                                int64_t rows,
                                                int64_t heads,
                                                void *stream);

} // namespace infinicore::op::deepseek_v4_fused_q_indexer_rope_hadamard_quant
