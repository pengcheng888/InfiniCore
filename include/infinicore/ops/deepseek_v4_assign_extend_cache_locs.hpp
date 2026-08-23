#pragma once

#include "common/op.hpp"

namespace infinicore::op {

void deepseek_v4_assign_extend_cache_locs_(const Tensor &req_pool_indices,
                                           const Tensor &req_to_token,
                                           const Tensor &start_offset,
                                           const Tensor &end_offset,
                                           Tensor out_cache_loc,
                                           int pool_len,
                                           int bs);

} // namespace infinicore::op
