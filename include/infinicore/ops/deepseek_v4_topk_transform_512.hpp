#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4TopkTransform512Kernel,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          Tensor,
                          int);

INFINICORE_GRAPH_OP_CLASS(DeepseekV4TopkTransform512SglangKernel,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          Tensor,
                          int);

void deepseek_v4_topk_transform_512_(const Tensor &scores,
                                     const Tensor &seq_lens,
                                     const Tensor &page_table,
                                     Tensor out_page_indices,
                                     int page_size);

void deepseek_v4_topk_transform_512_kernel_(const Tensor &scores,
                                            const Tensor &seq_lens,
                                            const Tensor &page_table,
                                            Tensor out_page_indices,
                                            int page_size);

void deepseek_v4_topk_transform_512_sglang_kernel_(const Tensor &scores,
                                                   const Tensor &seq_lens,
                                                   const Tensor &page_table,
                                                   Tensor out_page_indices,
                                                   int page_size);

} // namespace infinicore::op
