#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

namespace deepseek_v4 {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4C4CompressSglangStatefulKernel,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &);
INFINICORE_GRAPH_OP_CLASS(DeepseekV4C128CompressSglangStatefulKernel,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          Tensor,
                          const Tensor &,
                          const Tensor &);

} // namespace deepseek_v4

Tensor deepseek_v4_c4_compress_sglang_stateful_kernel(const Tensor &kv_score_input,
                                                      const Tensor &ape,
                                                      Tensor compressor_state,
                                                      const Tensor &write_loc,
                                                      const Tensor &extra_loc,
                                                      const Tensor &positions);

Tensor deepseek_v4_c4_compress_sglang_stateful(const Tensor &kv_score_input,
                                               const Tensor &ape,
                                               Tensor compressor_state,
                                               const Tensor &write_loc,
                                               const Tensor &extra_loc,
                                               const Tensor &positions);

Tensor deepseek_v4_c128_compress_sglang_stateful_kernel(const Tensor &kv_score_input,
                                                        const Tensor &ape,
                                                        Tensor compressor_state,
                                                        const Tensor &write_loc,
                                                        const Tensor &positions);

Tensor deepseek_v4_c128_compress_sglang_stateful(const Tensor &kv_score_input,
                                                 const Tensor &ape,
                                                 Tensor compressor_state,
                                                 const Tensor &write_loc,
                                                 const Tensor &positions);

} // namespace infinicore::op
