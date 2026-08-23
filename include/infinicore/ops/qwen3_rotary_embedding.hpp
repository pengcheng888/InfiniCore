#pragma once

#include "../device.hpp"
#include "common/op.hpp"

#include <optional>

namespace infinicore::op {

void qwen3_rotary_embedding_(const Tensor &positions,
                             Tensor query,
                             std::optional<Tensor> key,
                             int head_size,
                             const Tensor &cos_sin_cache,
                             bool is_neox);

} // namespace infinicore::op

