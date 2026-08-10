#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4C4PagedMqaWithTopkTransform512,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          Tensor,
                          int,
                          int,
                          bool);

void deepseek_v4_c4_paged_mqa_with_topk_transform_512_(const Tensor &q_fp8,
                                                       const Tensor &fused_weights,
                                                       const Tensor &indexer_kv_cache_raw,
                                                       const Tensor &c4_seq_lens,
                                                       const Tensor &page_table,
                                                       Tensor out_page_indices,
                                                       int max_c4_seq_len,
                                                       int page_size,
                                                       bool clean_logits);

} // namespace infinicore::op
