#pragma once

#include "common/op.hpp"

namespace infinicore::op {

void deepseek_v4_create_chunked_prefix_cache_kv_indices_(const Tensor &req_to_token,
                                                         const Tensor &req_pool_indices,
                                                         const Tensor &chunk_starts,
                                                         const Tensor &chunk_seq_lens,
                                                         const Tensor &chunk_cu_seq_lens,
                                                         Tensor chunk_kv_indices,
                                                         int col_num,
                                                         int bs);

} // namespace infinicore::op
