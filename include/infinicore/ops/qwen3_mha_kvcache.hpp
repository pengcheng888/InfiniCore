#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

Tensor qwen3_mha_kvcache(const Tensor &q,
                         const Tensor &k_cache,
                         const Tensor &v_cache,
                         const Tensor &seqlens_k,
                         const Tensor &block_table,
                         std::optional<Tensor> alibi_slopes,
                         float scale);

void qwen3_mha_kvcache_(Tensor out,
                        const Tensor &q,
                        const Tensor &k_cache,
                        const Tensor &v_cache,
                        const Tensor &seqlens_k,
                        const Tensor &block_table,
                        std::optional<Tensor> alibi_slopes,
                        float scale);

} // namespace infinicore::op
