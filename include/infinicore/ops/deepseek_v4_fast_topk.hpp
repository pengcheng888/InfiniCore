#pragma once

#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

void deepseek_v4_fast_topk_(const Tensor &score,
                            Tensor indices,
                            const Tensor &lengths,
                            std::optional<Tensor> row_starts);

void deepseek_v4_fast_topk_transform_fused_(const Tensor &score,
                                            const Tensor &lengths,
                                            Tensor dst_page_table,
                                            const Tensor &src_page_table,
                                            const Tensor &cu_seqlens_q,
                                            std::optional<Tensor> row_starts);

void deepseek_v4_fast_topk_transform_ragged_fused_(const Tensor &score,
                                                   const Tensor &lengths,
                                                   Tensor topk_indices_ragged,
                                                   const Tensor &topk_indices_offset,
                                                   std::optional<Tensor> row_starts);

} // namespace infinicore::op
