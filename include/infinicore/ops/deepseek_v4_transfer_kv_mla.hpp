#pragma once

#include "common/op.hpp"

namespace infinicore::op {

void deepseek_v4_transfer_kv_per_layer_mla_(const Tensor &src,
                                            Tensor dst,
                                            const Tensor &src_indices,
                                            const Tensor &dst_indices,
                                            int item_size,
                                            int block_quota,
                                            int num_warps_per_block);

void deepseek_v4_transfer_kv_per_layer_mla_pf_lf_(const Tensor &src,
                                                  Tensor dst,
                                                  const Tensor &src_indices,
                                                  const Tensor &dst_indices,
                                                  int layer_id,
                                                  int item_size,
                                                  int src_layout_dim,
                                                  int block_quota,
                                                  int num_warps_per_block);

} // namespace infinicore::op
