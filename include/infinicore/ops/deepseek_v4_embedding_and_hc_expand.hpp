#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

#include <cstdint>

namespace infinicore::op {

namespace deepseek_v4 {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4EmbeddingAndHcExpand, Tensor, const Tensor &, const Tensor &, int64_t);

} // namespace deepseek_v4

Tensor deepseek_v4_embedding_and_hc_expand(const Tensor &input, const Tensor &weight, int64_t hc_mult);
void deepseek_v4_embedding_and_hc_expand_(Tensor out, const Tensor &input, const Tensor &weight, int64_t hc_mult);
Tensor deepseek_v4_embedding_and_hc_expand_kernel(const Tensor &input, const Tensor &weight, int64_t hc_mult);
void deepseek_v4_embedding_and_hc_expand_kernel_(Tensor out, const Tensor &input, const Tensor &weight, int64_t hc_mult);
Tensor deepseek_v4_embedding_and_hc_expand_aten(const Tensor &input, const Tensor &weight, int64_t hc_mult);
void deepseek_v4_embedding_and_hc_expand_aten_(Tensor out, const Tensor &input, const Tensor &weight, int64_t hc_mult);

} // namespace infinicore::op
