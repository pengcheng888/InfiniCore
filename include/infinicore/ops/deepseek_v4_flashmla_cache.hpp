#pragma once

#include "common/op.hpp"

#include <optional>
#include <string>

namespace infinicore::op {

void deepseek_v4_fused_store_flashmla_cache_(const Tensor &kv_c,
                                             const Tensor &k_pe,
                                             Tensor kv_cache,
                                             const Tensor &slot_mapping,
                                             const std::string &kv_cache_dtype,
                                             const Tensor &scale);

void deepseek_v4_store_flashmla_raw_cache_(const Tensor &input,
                                           Tensor cache,
                                           const Tensor &indices,
                                           int page_size);

void deepseek_v4_flashmla_cache_indexer_(const Tensor &req_to_token,
                                         const Tensor &req_pool_indices,
                                         const Tensor &page_kernel_lens,
                                         std::optional<Tensor> kv_start_idx,
                                         Tensor kv_indices,
                                         int req_to_token_stride,
                                         int kv_indices_stride,
                                         int page_size);

} // namespace infinicore::op
