#include "infinicore/ops/deepseek_v4/linear_bf16_fp32.hpp"

#include "../graph_deferred.hpp"

#include <stdexcept>

namespace infinicore::op::deepseek_v4 {

Tensor linear_bf16_fp32(const Tensor &x, const Tensor &weight) {
    if (x->ndim() != 2 || weight->ndim() != 2) {
        throw std::runtime_error("deepseek_v4::linear_bf16_fp32 expects 2D input and weight tensors.");
    }
    if (x->size(1) != weight->size(1)) {
        throw std::runtime_error("deepseek_v4::linear_bf16_fp32 input/weight K dimension mismatch.");
    }
    auto out = Tensor::empty({x->size(0), weight->size(0)}, DataType::F32, x->device());
    linear_bf16_fp32_(out, x, weight);
    return out;
}

void linear_bf16_fp32_(Tensor out, const Tensor &x, const Tensor &weight) {
    auto out_graph = graph::GraphTensor(out);
    auto x_graph = graph::GraphTensor(x);
    auto weight_graph = graph::GraphTensor(weight);
    detail::record_or_run_host_graph_op([out_graph, x_graph, weight_graph]() mutable {
        linear_bf16_fp32_aten_(out_graph, x_graph, weight_graph);
    });
}

Tensor linear_bf16_fp32_aten(const Tensor &x, const Tensor &weight) {
    if (x->ndim() != 2 || weight->ndim() != 2) {
        throw std::runtime_error("deepseek_v4::linear_bf16_fp32_aten expects 2D input and weight tensors.");
    }
    if (x->size(1) != weight->size(1)) {
        throw std::runtime_error("deepseek_v4::linear_bf16_fp32_aten input/weight K dimension mismatch.");
    }
    auto out = Tensor::empty({x->size(0), weight->size(0)}, DataType::F32, x->device());
    linear_bf16_fp32_aten_(out, x, weight);
    return out;
}

} // namespace infinicore::op::deepseek_v4
