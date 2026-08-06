#pragma once

#include "common/op.hpp"

namespace infinicore::op {

void deepseek_v4_compressor_kv_score_packed_(Tensor out, const Tensor &x, const Tensor &wkv_gate);

void deepseek_v4_compressor_kv_score_unpacked_(Tensor out, const Tensor &x, const Tensor &wkv, const Tensor &wgate);

} // namespace infinicore::op
