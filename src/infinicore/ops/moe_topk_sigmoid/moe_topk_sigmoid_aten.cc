#include "infinicore/ops/moe_topk_sigmoid.hpp"

#include "infinicore/device.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ATen.h>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#elif defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

#include <optional>
#include <stdexcept>

namespace infinicore::op::moe_topk_sigmoid_impl::aten {

struct PlannedMeta {
    graph::GraphTensor topk_weights;
    graph::GraphTensor topk_indices;
    graph::GraphTensor gating_output;
    std::optional<graph::GraphTensor> correction_bias;
    bool renormalize;
};

void *plan(Tensor topk_weights,
           Tensor topk_indices,
           const Tensor &gating_output,
           const Tensor &correction_bias,
           const bool renormalize) {
    std::optional<graph::GraphTensor> correction_bias_graph;
    if (correction_bias) {
        correction_bias_graph.emplace(correction_bias);
    }
    return new PlannedMeta{
        graph::GraphTensor(topk_weights),
        graph::GraphTensor(topk_indices),
        graph::GraphTensor(gating_output),
        correction_bias_graph,
        renormalize};
}

void run(void *planned_meta) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API))
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    auto topk_weights = infinicore::adaptor::to_aten_tensor(planned->topk_weights);
    auto topk_indices = infinicore::adaptor::to_aten_tensor(planned->topk_indices);
    auto scores = infinicore::adaptor::to_aten_tensor(planned->gating_output).to(at::kFloat).sigmoid();
    if (planned->correction_bias.has_value()) {
        scores = scores + infinicore::adaptor::to_aten_tensor(*planned->correction_bias).to(at::kFloat);
    }
    const auto k = topk_weights.size(-1);
    auto topk = at::topk(scores, k, -1, true, true);
    auto values = std::get<0>(topk);
    auto indices = std::get<1>(topk);
    if (planned->renormalize) {
        values = values / values.sum(-1, true).clamp_min(1e-20);
    }
    topk_weights.copy_(values.to(topk_weights.scalar_type()));
    topk_indices.copy_(indices.to(topk_indices.scalar_type()));
#else
    (void)planned_meta;
    throw std::runtime_error("moe_topk_sigmoid ATen fallback requires an ATen-enabled HYGON/NVIDIA/METAX build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
#if defined(ENABLE_HYGON_API)
    MoeTopkSigmoid::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    MoeTopkSigmoid::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    MoeTopkSigmoid::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
#endif
#if defined(ENABLE_METAX_API)
    MoeTopkSigmoid::plan_dispatcher().registerDevice(Device::Type::METAX, &plan);
    MoeTopkSigmoid::run_dispatcher().registerDevice(Device::Type::METAX, &run);
    MoeTopkSigmoid::cleanup_dispatcher().registerDevice(Device::Type::METAX, &cleanup);
#endif
    return true;
}();

} // namespace infinicore::op::moe_topk_sigmoid_impl::aten
