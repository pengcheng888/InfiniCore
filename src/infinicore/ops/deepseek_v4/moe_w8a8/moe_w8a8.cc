#include "infinicore/ops/deepseek_v4/moe_w8a8.hpp"

#include "../graph_deferred.hpp"

namespace infinicore::op::deepseek_v4 {

void moe_w8a8_(Tensor y,
               const Tensor &x,
               const Tensor &topk_weights,
               const Tensor &topk_indices,
               const Tensor &w13,
               const Tensor &w13_scale,
               const Tensor &w2,
               const Tensor &w2_scale,
               double swiglu_limit) {
    auto y_graph = graph::GraphTensor(y);
    auto x_graph = graph::GraphTensor(x);
    auto topk_weights_graph = graph::GraphTensor(topk_weights);
    auto topk_indices_graph = graph::GraphTensor(topk_indices);
    auto w13_graph = graph::GraphTensor(w13);
    auto w13_scale_graph = graph::GraphTensor(w13_scale);
    auto w2_graph = graph::GraphTensor(w2);
    auto w2_scale_graph = graph::GraphTensor(w2_scale);
    detail::record_or_run_host_graph_op([y_graph,
                                         x_graph,
                                         topk_weights_graph,
                                         topk_indices_graph,
                                         w13_graph,
                                         w13_scale_graph,
                                         w2_graph,
                                         w2_scale_graph,
                                         swiglu_limit]() mutable {
        moe_w8a8_aten_(y_graph,
                       x_graph,
                       topk_weights_graph,
                       topk_indices_graph,
                       w13_graph,
                       w13_scale_graph,
                       w2_graph,
                       w2_scale_graph,
                       swiglu_limit);
    });
}

} // namespace infinicore::op::deepseek_v4
