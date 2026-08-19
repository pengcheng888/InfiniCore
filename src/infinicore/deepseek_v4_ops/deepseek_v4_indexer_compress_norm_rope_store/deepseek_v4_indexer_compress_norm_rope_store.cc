#include "infinicore/ops/deepseek_v4_indexer_compress_norm_rope_store.hpp"

namespace infinicore::op {

void deepseek_v4_indexer_compress_norm_rope_store_(const Tensor &kv,
                                                   const Tensor &norm_weight,
                                                   float epsilon,
                                                   const Tensor &freqs_cis,
                                                   const Tensor &positions,
                                                   const Tensor &out_loc,
                                                   Tensor kvcache,
                                                   int page_size) {
    deepseek_v4_indexer_compress_norm_rope_store_kernel_(kv,
                                                         norm_weight,
                                                         epsilon,
                                                         freqs_cis,
                                                         positions,
                                                         out_loc,
                                                         kvcache,
                                                         page_size);
}

} // namespace infinicore::op
