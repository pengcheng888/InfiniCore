#pragma once

#include "common/op.hpp"

namespace infinicore::op {

void deepseek_v4_transfer_kv_per_layer_(const Tensor &src_k,
                                        Tensor dst_k,
                                        const Tensor &src_v,
                                        Tensor dst_v,
                                        const Tensor &src_indices,
                                        const Tensor &dst_indices,
                                        int item_size,
                                        int block_quota,
                                        int num_warps_per_block);

void deepseek_v4_transfer_kv_per_layer_pf_lf_(const Tensor &src_k,
                                              Tensor dst_k,
                                              const Tensor &src_v,
                                              Tensor dst_v,
                                              const Tensor &src_indices,
                                              const Tensor &dst_indices,
                                              int layer_id,
                                              int item_size,
                                              int src_layout_dim,
                                              int block_quota,
                                              int num_warps_per_block);

} // namespace infinicore::op
