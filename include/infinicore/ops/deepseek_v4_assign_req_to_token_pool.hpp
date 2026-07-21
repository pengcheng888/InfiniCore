#pragma once

#include "common/op.hpp"

namespace infinicore::op {

void deepseek_v4_assign_req_to_token_pool_(const Tensor &req_pool_indices,
                                           Tensor req_to_token,
                                           const Tensor &allocate_lens,
                                           const Tensor &new_allocate_lens,
                                           Tensor out_cache_loc,
                                           int shape,
                                           int bs);

} // namespace infinicore::op
