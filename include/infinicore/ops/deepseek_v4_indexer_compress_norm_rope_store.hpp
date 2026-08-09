#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4IndexerCompressNormRopeStoreKernel,
                          const Tensor &,
                          const Tensor &,
                          float,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          Tensor,
                          int);

void deepseek_v4_indexer_compress_norm_rope_store_kernel_(const Tensor &kv,
                                                          const Tensor &norm_weight,
                                                          float epsilon,
                                                          const Tensor &freqs_cis,
                                                          const Tensor &positions,
                                                          const Tensor &out_loc,
                                                          Tensor kvcache,
                                                          int page_size);

void deepseek_v4_indexer_compress_norm_rope_store_(const Tensor &kv,
                                                   const Tensor &norm_weight,
                                                   float epsilon,
                                                   const Tensor &freqs_cis,
                                                   const Tensor &positions,
                                                   const Tensor &out_loc,
                                                   Tensor kvcache,
                                                   int page_size);

} // namespace infinicore::op
