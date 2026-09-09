#pragma once

#include "../common/op.hpp"

#include <optional>

namespace infinicore::op::deepseek_v4 {

void fused_rope_(Tensor query,
                 std::optional<Tensor> key,
                 const Tensor &freqs_cis,
                 const Tensor &positions,
                 bool inverse);

void fused_rope_aten_(Tensor query,
                      std::optional<Tensor> key,
                      const Tensor &freqs_cis,
                      const Tensor &positions,
                      bool inverse);

} // namespace infinicore::op::deepseek_v4
