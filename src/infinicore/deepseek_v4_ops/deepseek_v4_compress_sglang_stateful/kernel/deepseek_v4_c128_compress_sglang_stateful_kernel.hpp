#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_c128_compress_sglang_stateful_kernel_impl {

void launch_c128_compress_stateful_sglang(void *output,
                                          int output_dtype,
                                          const void *kv_score,
                                          int kv_score_dtype,
                                          void *compressor_state,
                                          int state_dtype,
                                          const void *ape,
                                          int ape_dtype,
                                          const void *write_loc,
                                          bool write_loc_i64,
                                          const void *positions,
                                          bool positions_i64,
                                          int64_t tokens,
                                          int64_t head_dim,
                                          void *stream);

} // namespace infinicore::op::deepseek_v4_c128_compress_sglang_stateful_kernel_impl
