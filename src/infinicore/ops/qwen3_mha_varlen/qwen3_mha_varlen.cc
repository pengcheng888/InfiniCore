#include "infinicore/ops/qwen3_mha_varlen.hpp"

#include "infinicore/ops/mha_varlen.hpp"

namespace infinicore::op {

Tensor qwen3_mha_varlen(const Tensor &q,
                        const Tensor &k,
                        const Tensor &v,
                        const Tensor &cum_seqlens_q,
                        const Tensor &cum_seqlens_k,
                        std::optional<Tensor> block_table,
                        int max_seqlen_q,
                        int max_seqlen_k,
                        std::optional<Tensor> alibi_slopes,
                        float scale) {
    return mha_varlen(q, k, v, cum_seqlens_q, cum_seqlens_k, block_table, max_seqlen_q, max_seqlen_k, alibi_slopes, scale);
}

void qwen3_mha_varlen_(Tensor out,
                       const Tensor &q,
                       const Tensor &k,
                       const Tensor &v,
                       const Tensor &cum_seqlens_q,
                       const Tensor &cum_seqlens_k,
                       std::optional<Tensor> block_table,
                       int max_seqlen_q,
                       int max_seqlen_k,
                       std::optional<Tensor> alibi_slopes,
                       float scale) {
    mha_varlen_(out, q, k, v, cum_seqlens_q, cum_seqlens_k, block_table, max_seqlen_q, max_seqlen_k, alibi_slopes, scale);
}

} // namespace infinicore::op
