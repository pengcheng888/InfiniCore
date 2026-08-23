#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

#include <optional>

namespace infinicore::op {

namespace deepseek_v4 {

INFINICORE_GRAPH_OP_CLASS(FusedRope, Tensor, std::optional<Tensor>, const Tensor &, const Tensor &, bool);

} // namespace deepseek_v4

void deepseek_v4_fused_rope_aten_(Tensor query,
                                  std::optional<Tensor> key,
                                  const Tensor &freqs_cis,
                                  const Tensor &positions,
                                  bool inverse);

void deepseek_v4_fused_rope_kernel_(Tensor query,
                                    std::optional<Tensor> key,
                                    const Tensor &freqs_cis,
                                    const Tensor &positions,
                                    bool inverse);

void deepseek_v4_fused_rope_(Tensor query,
                             std::optional<Tensor> key,
                             const Tensor &freqs_cis,
                             const Tensor &positions,
                             bool inverse);

} // namespace infinicore::op
