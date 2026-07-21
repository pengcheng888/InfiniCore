#pragma once

#include "../device.hpp"
#include "common/op.hpp"

#include <optional>

namespace infinicore::op {

void deepseek_v4_create_flashmla_kv_indices_(const Tensor &req_to_token,
                                             const Tensor &req_pool_indices,
                                             const Tensor &page_kernel_lens,
                                             std::optional<Tensor> kv_start_idx,
                                             Tensor kv_indices,
                                             int req_to_token_stride,
                                             int kv_indices_stride,
                                             int page_size);

} // namespace infinicore::op
