#pragma once

#include "common/op.hpp"
#include <string>

namespace infinicore::op {

void deepseek_v4_concat_and_cache_mla_(const Tensor &kv_c,
                                       const Tensor &k_pe,
                                       Tensor kv_cache,
                                       const Tensor &slot_mapping,
                                       const std::string &kv_cache_dtype,
                                       const Tensor &scale);

} // namespace infinicore::op
