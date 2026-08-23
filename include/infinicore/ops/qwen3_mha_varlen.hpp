#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <optional>

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
                        float scale);

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
                       float scale);

} // namespace infinicore::op
