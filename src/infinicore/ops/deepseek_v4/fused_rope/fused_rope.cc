#include "infinicore/ops/deepseek_v4/fused_rope.hpp"

#include "../graph_deferred.hpp"

namespace infinicore::op::deepseek_v4 {

void fused_rope_(Tensor query,
                 std::optional<Tensor> key,
                 const Tensor &freqs_cis,
                 const Tensor &positions,
                 bool inverse) {
    auto query_graph = graph::GraphTensor(query);
    std::optional<Tensor> key_graph = std::nullopt;
    if (key.has_value() && key.value()) {
        key_graph = graph::GraphTensor(key.value());
    }
    auto freqs_cis_graph = graph::GraphTensor(freqs_cis);
    auto positions_graph = graph::GraphTensor(positions);
    detail::record_or_run_host_graph_op([query_graph,
                                         key_graph,
                                         freqs_cis_graph,
                                         positions_graph,
                                         inverse]() mutable {
        fused_rope_aten_(query_graph, key_graph, freqs_cis_graph, positions_graph, inverse);
    });
}

} // namespace infinicore::op::deepseek_v4
