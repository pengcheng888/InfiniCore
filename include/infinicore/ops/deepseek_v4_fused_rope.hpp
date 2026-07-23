#pragma once

#include "common/op.hpp"

#include <optional>

namespace infinicore::op {

void deepseek_v4_fused_rope_(Tensor query,
                             std::optional<Tensor> key,
                             const Tensor &freqs_cis,
                             const Tensor &positions,
                             bool inverse);

} // namespace infinicore::op
