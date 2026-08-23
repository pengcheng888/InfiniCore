#include "infinicore/ops/deepseek_v4_fused_rope.hpp"

namespace infinicore::op {

void deepseek_v4_fused_rope_(Tensor query,
                             std::optional<Tensor> key,
                             const Tensor &freqs_cis,
                             const Tensor &positions,
                             bool inverse) {
    deepseek_v4_fused_rope_kernel_(query, key, freqs_cis, positions, inverse);
}

} // namespace infinicore::op
