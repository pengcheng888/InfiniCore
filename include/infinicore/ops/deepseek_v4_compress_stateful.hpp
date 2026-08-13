#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4C4CompressStatefulKernel,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &);
INFINICORE_GRAPH_OP_CLASS(DeepseekV4C128CompressStatefulKernel,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          Tensor,
                          const Tensor &,
                          const Tensor &);

Tensor deepseek_v4_c4_compress_stateful_naive(const Tensor &kv_score_input,
                                              const Tensor &ape,
                                              Tensor compressor_state,
                                              const Tensor &write_loc,
                                              const Tensor &extra_loc,
                                              const Tensor &positions);

Tensor deepseek_v4_c4_compress_stateful_kernel(const Tensor &kv_score_input,
                                               const Tensor &ape,
                                               Tensor compressor_state,
                                               const Tensor &write_loc,
                                               const Tensor &extra_loc,
                                               const Tensor &positions);

Tensor deepseek_v4_c4_compress_stateful(const Tensor &kv_score_input,
                                        const Tensor &ape,
                                        Tensor compressor_state,
                                        const Tensor &write_loc,
                                        const Tensor &extra_loc,
                                        const Tensor &positions);

Tensor deepseek_v4_c128_compress_stateful_naive(const Tensor &kv_score_input,
                                                const Tensor &ape,
                                                Tensor compressor_state,
                                                const Tensor &write_loc,
                                                const Tensor &positions);

Tensor deepseek_v4_c128_compress_stateful_kernel(const Tensor &kv_score_input,
                                                 const Tensor &ape,
                                                 Tensor compressor_state,
                                                 const Tensor &write_loc,
                                                 const Tensor &positions);

Tensor deepseek_v4_c128_compress_stateful(const Tensor &kv_score_input,
                                          const Tensor &ape,
                                          Tensor compressor_state,
                                          const Tensor &write_loc,
                                          const Tensor &positions);

} // namespace infinicore::op
