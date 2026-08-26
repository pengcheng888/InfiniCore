#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

#include <string>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(ConcatAndCacheMla,
                          const Tensor &,
                          const Tensor &,
                          Tensor,
                          const Tensor &,
                          const std::string &,
                          Tensor);

void concat_and_cache_mla_(const Tensor &kv_c,
                           const Tensor &k_pe,
                           Tensor kv_cache,
                           const Tensor &slot_mapping,
                           const std::string &kv_cache_dtype,
                           Tensor scale);

} // namespace infinicore::op
