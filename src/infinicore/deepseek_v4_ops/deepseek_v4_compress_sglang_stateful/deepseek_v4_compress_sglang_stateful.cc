#include "infinicore/ops/deepseek_v4_compress_sglang_stateful.hpp"

namespace infinicore::op {

Tensor deepseek_v4_c4_compress_sglang_stateful(const Tensor &kv_score_input,
                                               const Tensor &ape,
                                               Tensor compressor_state,
                                               const Tensor &write_loc,
                                               const Tensor &extra_loc,
                                               const Tensor &positions) {
    return deepseek_v4_c4_compress_sglang_stateful_kernel(kv_score_input,
                                                          ape,
                                                          compressor_state,
                                                          write_loc,
                                                          extra_loc,
                                                          positions);
}

Tensor deepseek_v4_c128_compress_sglang_stateful(const Tensor &kv_score_input,
                                                 const Tensor &ape,
                                                 Tensor compressor_state,
                                                 const Tensor &write_loc,
                                                 const Tensor &positions) {
    return deepseek_v4_c128_compress_sglang_stateful_kernel(kv_score_input,
                                                            ape,
                                                            compressor_state,
                                                            write_loc,
                                                            positions);
}

} // namespace infinicore::op
