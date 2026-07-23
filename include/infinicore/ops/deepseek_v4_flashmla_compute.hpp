#pragma once

#include "common/op.hpp"

#include <optional>

namespace infinicore::op {


Tensor deepseek_v4_c4_compress_prefill_reference(const Tensor &kv_score_input,
                                                 const Tensor &ape);

Tensor deepseek_v4_c4_compress_stateful_reference(const Tensor &kv_score_input,
                                                  const Tensor &ape,
                                                  Tensor compressor_state,
                                                  const Tensor &write_loc,
                                                  const Tensor &extra_loc,
                                                  const Tensor &positions);

Tensor deepseek_v4_c128_compress_stateful_reference(const Tensor &kv_score_input,
                                                    const Tensor &ape,
                                                    Tensor compressor_state,
                                                    const Tensor &write_loc,
                                                    const Tensor &positions);

void deepseek_v4_flashmla_sparse_attention_(const Tensor &q,
                                            const Tensor &raw_cache,
                                            const Tensor &indices,
                                            const Tensor &topk_lengths,
                                            std::optional<Tensor> attn_sink,
                                            Tensor output,
                                            float softmax_scale,
                                            int page_size,
                                            int head_dim_v,
                                            std::optional<Tensor> extra_raw_cache = std::nullopt,
                                            std::optional<Tensor> extra_indices = std::nullopt,
                                            std::optional<Tensor> extra_topk_lengths = std::nullopt,
                                            int extra_page_size = 0);

} // namespace infinicore::op
