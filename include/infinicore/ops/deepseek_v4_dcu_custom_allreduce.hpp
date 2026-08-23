#pragma once

#include "../tensor.hpp"

namespace infinicore::op {

bool deepseek_v4_dcu_custom_allreduce_(Tensor output,
                                       const Tensor &input,
                                       int tp_rank,
                                       int tp_size,
                                       int max_size_bytes = 8192 * 512);

} // namespace infinicore::op
