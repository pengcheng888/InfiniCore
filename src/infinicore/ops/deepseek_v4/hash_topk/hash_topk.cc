#include "infinicore/ops/deepseek_v4/hash_topk.hpp"

#include "../graph_deferred.hpp"

namespace infinicore::op::deepseek_v4 {

void hash_topk_(Tensor topk_weights,
                Tensor topk_indices,
                const Tensor &router_logits,
                const Tensor &input_ids,
                const Tensor &tid2eid,
                int64_t num_fused_shared_experts,
                float routed_scaling_factor,
                const std::string &scoring_func) {
    auto topk_weights_graph = graph::GraphTensor(topk_weights);
    auto topk_indices_graph = graph::GraphTensor(topk_indices);
    auto router_logits_graph = graph::GraphTensor(router_logits);
    auto input_ids_graph = graph::GraphTensor(input_ids);
    auto tid2eid_graph = graph::GraphTensor(tid2eid);
    detail::record_or_run_host_graph_op([topk_weights_graph,
                                         topk_indices_graph,
                                         router_logits_graph,
                                         input_ids_graph,
                                         tid2eid_graph,
                                         num_fused_shared_experts,
                                         routed_scaling_factor,
                                         scoring_func]() mutable {
        hash_topk_aten_(topk_weights_graph,
                        topk_indices_graph,
                        router_logits_graph,
                        input_ids_graph,
                        tid2eid_graph,
                        num_fused_shared_experts,
                        routed_scaling_factor,
                        scoring_func);
    });
}

} // namespace infinicore::op::deepseek_v4
