#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

void qwen3_fused_qk_norm_rope_(Tensor qkv,
                               int num_heads_q,
                               int num_heads_k,
                               int num_heads_v,
                               int head_dim,
                               float eps,
                               const Tensor &q_weight,
                               const Tensor &k_weight,
                               float base,
                               bool is_neox,
                               const Tensor &position_ids,
                               float factor,
                               float low,
                               float high,
                               float attention_factor,
                               int rotary_dim);

} // namespace infinicore::op

