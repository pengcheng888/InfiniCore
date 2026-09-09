#include "infinicore/ops/deepseek_v4/biased_topk.hpp"

#include "../graph_deferred.hpp"

namespace infinicore::op::deepseek_v4 {

void topk_(Tensor topk_weights,
           Tensor topk_indices,
           const Tensor &router_logits,
           const Tensor &correction_bias,
           bool renormalize,
           const std::string &scoring_func) {
    auto topk_weights_graph = graph::GraphTensor(topk_weights);
    auto topk_indices_graph = graph::GraphTensor(topk_indices);
    auto router_logits_graph = graph::GraphTensor(router_logits);
    auto correction_bias_graph = graph::GraphTensor(correction_bias);
    detail::record_or_run_host_graph_op([topk_weights_graph,
                                         topk_indices_graph,
                                         router_logits_graph,
                                         correction_bias_graph,
                                         renormalize,
                                         scoring_func]() mutable {
        topk_aten_(topk_weights_graph,
                   topk_indices_graph,
                   router_logits_graph,
                   correction_bias_graph,
                   renormalize,
                   scoring_func);
    });
}

} // namespace infinicore::op::deepseek_v4
