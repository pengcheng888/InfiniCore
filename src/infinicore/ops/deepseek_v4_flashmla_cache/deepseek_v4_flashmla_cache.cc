#include "infinicore/ops/deepseek_v4_flashmla_cache.hpp"

#include "infinicore/ops/deepseek_v4_concat_and_cache_mla.hpp"
#include "infinicore/ops/deepseek_v4_create_flashmla_kv_indices.hpp"

namespace infinicore::op {

void deepseek_v4_fused_store_flashmla_cache_(const Tensor &kv_c,
                                             const Tensor &k_pe,
                                             Tensor kv_cache,
                                             const Tensor &slot_mapping,
                                             const std::string &kv_cache_dtype,
                                             const Tensor &scale) {
    deepseek_v4_concat_and_cache_mla_(kv_c,
                                      k_pe,
                                      kv_cache,
                                      slot_mapping,
                                      kv_cache_dtype,
                                      scale);
}

void deepseek_v4_flashmla_cache_indexer_(const Tensor &req_to_token,
                                         const Tensor &req_pool_indices,
                                         const Tensor &page_kernel_lens,
                                         std::optional<Tensor> kv_start_idx,
                                         Tensor kv_indices,
                                         int req_to_token_stride,
                                         int kv_indices_stride,
                                         int page_size) {
    deepseek_v4_create_flashmla_kv_indices_(req_to_token,
                                            req_pool_indices,
                                            page_kernel_lens,
                                            kv_start_idx,
                                            kv_indices,
                                            req_to_token_stride,
                                            kv_indices_stride,
                                            page_size);
}

} // namespace infinicore::op
