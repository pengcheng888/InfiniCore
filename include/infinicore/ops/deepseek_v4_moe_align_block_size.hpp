#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

void deepseek_v4_moe_align_block_size_(const Tensor &topk_ids,
                                       int num_experts,
                                       int block_size,
                                       Tensor sorted_token_ids,
                                       Tensor experts_ids,
                                       Tensor num_tokens_post_pad,
                                       Tensor cumsum_buffer,
                                       bool pad_sorted_token_ids);

} // namespace infinicore::op
