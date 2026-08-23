#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_fused_norm_rope_inplace_native {

void launch_fused_norm_rope_inplace(void *input,
                                    const void *norm_weight,
                                    const float *freqs_cis,
                                    const void *positions,
                                    bool positions_i64,
                                    int64_t tokens,
                                    float epsilon,
                                    void *stream);

} // namespace infinicore::op::deepseek_v4_fused_norm_rope_inplace_native
