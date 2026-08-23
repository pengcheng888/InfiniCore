#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_shared_experts_impl_int8_marlin {

void launch_fill_single_expert_metadata(void *sorted_token_ids,
                                        void *expert_ids,
                                        void *num_tokens_post_pad,
                                        void *topk_weights,
                                        int64_t tokens,
                                        int top_k,
                                        int block_size,
                                        void *stream);

} // namespace infinicore::op::deepseek_v4_shared_experts_impl_int8_marlin
