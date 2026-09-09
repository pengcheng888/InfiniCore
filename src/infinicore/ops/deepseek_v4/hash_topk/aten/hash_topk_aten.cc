#include "infinicore/ops/deepseek_v4/hash_topk.hpp"

#include "infinicore/device.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ATen.h>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#elif defined(ENABLE_NVIDIA_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

#include <stdexcept>
#include <string>

namespace infinicore::op::deepseek_v4 {
namespace {

void check_accelerator_tensor(const Tensor &tensor, const char *op_name) {
#if defined(ENABLE_HYGON_API)
    if (tensor->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error(std::string(op_name) + " expects HYGON tensors in this build.");
    }
#elif defined(ENABLE_NVIDIA_API)
    if (tensor->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error(std::string(op_name) + " expects NVIDIA tensors in this build.");
    }
#else
    (void)tensor;
    (void)op_name;
#endif
}

at::Tensor router_scores(const at::Tensor &logits, const std::string &scoring_func) {
    if (scoring_func == "softmax") {
        return at::softmax(logits, -1);
    }
    if (scoring_func == "sigmoid") {
        return at::sigmoid(logits);
    }
    if (scoring_func == "sqrtsoftplus") {
        return at::sqrt(at::softplus(logits));
    }
    throw std::runtime_error("deepseek_v4::hash_topk_aten_ unsupported scoring_func: " + scoring_func);
}

} // namespace

void hash_topk_aten_(Tensor topk_weights,
                     Tensor topk_indices,
                     const Tensor &router_logits,
                     const Tensor &input_ids,
                     const Tensor &tid2eid,
                     int64_t num_fused_shared_experts,
                     float routed_scaling_factor,
                     const std::string &scoring_func) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_accelerator_tensor(router_logits, "deepseek_v4::hash_topk_aten_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    if (router_logits->ndim() != 2 || input_ids->ndim() != 1 || tid2eid->ndim() != 2 ||
        topk_weights->ndim() != 2 || topk_indices->ndim() != 2) {
        throw std::runtime_error("deepseek_v4::hash_topk_aten_ expects logits [tokens, experts], input_ids [tokens], tid2eid [vocab, topk].");
    }
    const auto tokens = router_logits->size(0);
    const auto routed_topk = tid2eid->size(1);
    const auto total_topk = routed_topk + static_cast<size_t>(num_fused_shared_experts);
    if (input_ids->size(0) != tokens || topk_weights->shape() != Shape{tokens, total_topk} ||
        topk_indices->shape() != Shape{tokens, total_topk}) {
        throw std::runtime_error("deepseek_v4::hash_topk_aten_ topk shape mismatch.");
    }

    auto weights_at = infinicore::adaptor::to_aten_tensor(topk_weights);
    auto indices_at = infinicore::adaptor::to_aten_tensor(topk_indices);
    auto logits_at = infinicore::adaptor::to_aten_tensor(router_logits);
    auto input_ids_at = infinicore::adaptor::to_aten_tensor(input_ids).to(at::kLong);
    auto tid2eid_at = infinicore::adaptor::to_aten_tensor(tid2eid).to(at::kLong);

    auto scores = router_scores(logits_at, scoring_func);
    auto routed_ids = tid2eid_at.index_select(0, input_ids_at);
    auto routed_weights = scores.gather(1, routed_ids);
    if (scoring_func != "softmax") {
        routed_weights = routed_weights / routed_weights.sum(-1, true);
    }

    if (num_fused_shared_experts == 0) {
        indices_at.copy_(routed_ids.to(indices_at.scalar_type()));
        weights_at.copy_(routed_weights.to(weights_at.scalar_type()));
        return;
    }

    auto out_ids = at::empty({static_cast<int64_t>(tokens), static_cast<int64_t>(total_topk)},
                             indices_at.options().dtype(at::kLong));
    auto out_weights = at::zeros({static_cast<int64_t>(tokens), static_cast<int64_t>(total_topk)},
                                 weights_at.options().dtype(at::kFloat));
    out_ids.slice(1, 0, routed_topk).copy_(routed_ids);
    out_weights.slice(1, 0, routed_topk).copy_(routed_weights);
    auto fused_ids = at::randint(router_logits->size(1),
                                 router_logits->size(1) + num_fused_shared_experts,
                                 {static_cast<int64_t>(tokens)},
                                 out_ids.options());
    out_ids.slice(1, routed_topk, total_topk).copy_(fused_ids.unsqueeze(1));
    out_weights.slice(1, routed_topk, total_topk).copy_(
        (routed_weights.sum(-1) / static_cast<double>(routed_scaling_factor)).unsqueeze(1));
    indices_at.copy_(out_ids.to(indices_at.scalar_type()));
    weights_at.copy_(out_weights.to(weights_at.scalar_type()));
#else
    (void)topk_weights;
    (void)topk_indices;
    (void)router_logits;
    (void)input_ids;
    (void)tid2eid;
    (void)num_fused_shared_experts;
    (void)routed_scaling_factor;
    (void)scoring_func;
    throw std::runtime_error("deepseek_v4::hash_topk_aten_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op::deepseek_v4
