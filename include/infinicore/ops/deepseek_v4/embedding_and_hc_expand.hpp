#pragma once

#include "../../device.hpp"
#include "../../graph/graph.hpp"
#include "../common/op.hpp"

#include <cstdint>

namespace infinicore::op::deepseek_v4 {


INFINICORE_GRAPH_OP_CLASS(EmbeddingAndHcExpand, Tensor, const Tensor &, const Tensor &, int64_t);


Tensor embedding_and_hc_expand(const Tensor &input, const Tensor &weight, int64_t hc_mult);
void embedding_and_hc_expand_(Tensor out, const Tensor &input, const Tensor &weight, int64_t hc_mult);
Tensor embedding_and_hc_expand_kernel(const Tensor &input, const Tensor &weight, int64_t hc_mult);
void embedding_and_hc_expand_kernel_(Tensor out, const Tensor &input, const Tensor &weight, int64_t hc_mult);
Tensor embedding_and_hc_expand_aten(const Tensor &input, const Tensor &weight, int64_t hc_mult);
void embedding_and_hc_expand_aten_(Tensor out, const Tensor &input, const Tensor &weight, int64_t hc_mult);

} // namespace infinicore::op
