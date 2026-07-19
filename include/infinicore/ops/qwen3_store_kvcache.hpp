#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

void qwen3_store_kvcache_(const Tensor &k,
                          const Tensor &v,
                          Tensor k_cache,
                          Tensor v_cache,
                          const Tensor &indices);

} // namespace infinicore::op

