#include "infinicore/ops/qwen3_mha_kvcache.hpp"

#include "infinicore/ops/mha_kvcache.hpp"

namespace infinicore::op {

Tensor qwen3_mha_kvcache(const Tensor &q,
                         const Tensor &k_cache,
                         const Tensor &v_cache,
                         const Tensor &seqlens_k,
                         const Tensor &block_table,
                         std::optional<Tensor> alibi_slopes,
                         float scale) {
    return mha_kvcache(q, k_cache, v_cache, seqlens_k, block_table, alibi_slopes, scale);
}

void qwen3_mha_kvcache_(Tensor out,
                        const Tensor &q,
                        const Tensor &k_cache,
                        const Tensor &v_cache,
                        const Tensor &seqlens_k,
                        const Tensor &block_table,
                        std::optional<Tensor> alibi_slopes,
                        float scale) {
    mha_kvcache_(out, q, k_cache, v_cache, seqlens_k, block_table, alibi_slopes, scale);
}

} // namespace infinicore::op
