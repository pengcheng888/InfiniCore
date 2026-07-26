#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_flashmla_compute_kernel {

enum Dsv4ScalarType : int {
    kDsv4BF16 = 0,
    kDsv4F16 = 1,
    kDsv4F32 = 2,
};

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

void launch_c4_compress_stateful(void *output,
                                  int output_dtype,
                                  const void *kv_score,
                                  int kv_score_dtype,
                                  void *compressor_state,
                                  int state_dtype,
                                  const void *ape,
                                  int ape_dtype,
                                  int ape_layout,
                                  const void *write_loc,
                                  bool write_loc_i64,
                                  const void *extra_loc,
                                  bool extra_loc_i64,
                                  int64_t extra_cols,
                                  const void *positions,
                                  bool positions_i64,
                                  int64_t tokens,
                                  int64_t head_dim,
                                  void *stream);

void launch_c128_compress_stateful(void *output,
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

} // namespace infinicore::op::deepseek_v4_flashmla_compute_kernel
