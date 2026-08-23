#include "infinicore/ops/deepseek_v4_topk_transform_512.hpp"

namespace infinicore::op {

void deepseek_v4_topk_transform_512_(const Tensor &scores,
                                     const Tensor &seq_lens,
                                     const Tensor &page_table,
                                     Tensor out_page_indices,
                                     int page_size) {
    deepseek_v4_topk_transform_512_sglang_kernel_(scores,
                                                  seq_lens,
                                                  page_table,
                                                  out_page_indices,
                                                  page_size);
}

} // namespace infinicore::op
