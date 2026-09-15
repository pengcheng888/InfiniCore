#include "infinicore/ops/deepseek_v4/fused_rope.hpp"

namespace infinicore::op::deepseek_v4 {

void fused_rope_(Tensor query,
                 std::optional<Tensor> key,
                 const Tensor &freqs_cis,
                 const Tensor &positions,
                 bool inverse) {
    fused_rope_kernel_(query, key, freqs_cis, positions, inverse);
}

} // namespace infinicore::op::deepseek_v4
