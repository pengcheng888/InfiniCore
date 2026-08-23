#pragma once

#include "common/op.hpp"

namespace infinicore::op {

void deepseek_v4_fused_qk_norm_rope_(Tensor qkv,
                                     int num_heads_q,
                                     int num_heads_k,
                                     int num_heads_v,
                                     int head_dim,
                                     float eps,
                                     const Tensor &q_weight,
                                     const Tensor &k_weight,
                                     const Tensor &cos_sin_cache,
                                     bool is_neox,
                                     const Tensor &position_ids);

} // namespace infinicore::op
