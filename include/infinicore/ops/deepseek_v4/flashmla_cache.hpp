#pragma once

#include "../../device.hpp"
#include "../../tensor.hpp"

namespace infinicore::op::deepseek_v4 {

void store_flashmla_raw_cache_(const Tensor &input,
                                           Tensor cache,
                                           const Tensor &indices,
                                           int page_size);

} // namespace infinicore::op::deepseek_v4
