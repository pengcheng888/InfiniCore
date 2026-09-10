#include "infinicore/ops/deepseek_v4/biased_topk.hpp"

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
    if (scoring_func == "sigmoid") {
        return at::sigmoid(logits);
    }
    if (scoring_func == "sqrtsoftplus") {
        return at::sqrt(at::softplus(logits));
    }
    throw std::runtime_error("deepseek_v4::topk_aten_ unsupported scoring_func: " + scoring_func);
}

} // namespace

void topk_aten_(Tensor topk_weights,
                Tensor topk_indices,
                const Tensor &router_logits,
                const Tensor &correction_bias,
                bool renormalize,
                const std::string &scoring_func) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_accelerator_tensor(router_logits, "deepseek_v4::topk_aten_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    if (router_logits->ndim() != 2 || correction_bias->ndim() != 1 ||
        topk_weights->ndim() != 2 || topk_indices->ndim() != 2) {
        throw std::runtime_error("deepseek_v4::topk_aten_ expects logits [tokens, experts] and bias [experts].");
    }
    const auto tokens = router_logits->size(0);
    const auto experts = router_logits->size(1);
    const auto topk = topk_weights->size(1);
    if (correction_bias->size(0) != experts || topk_indices->shape() != topk_weights->shape() ||
        topk_weights->shape() != Shape{tokens, topk}) {
        throw std::runtime_error("deepseek_v4::topk_aten_ topk shape mismatch.");
    }

    auto weights_at = infinicore::adaptor::to_aten_tensor(topk_weights);
    auto indices_at = infinicore::adaptor::to_aten_tensor(topk_indices);
    auto logits_at = infinicore::adaptor::to_aten_tensor(router_logits);
    auto bias_at = infinicore::adaptor::to_aten_tensor(correction_bias).to(at::kFloat);

    auto scores = router_scores(logits_at, scoring_func);
    auto choice = scores + bias_at.unsqueeze(0);
    auto topk_result = at::topk(choice, static_cast<int64_t>(topk), -1, true, false);
    auto ids = std::get<1>(topk_result);
    auto selected = scores.gather(1, ids);
    if (renormalize) {
        selected = selected / selected.sum(-1, true);
    }
    indices_at.copy_(ids.to(indices_at.scalar_type()));
    weights_at.copy_(selected.to(weights_at.scalar_type()));
#else
    (void)topk_weights;
    (void)topk_indices;
    (void)router_logits;
    (void)correction_bias;
    (void)renormalize;
    (void)scoring_func;
    throw std::runtime_error("deepseek_v4::topk_aten_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op::deepseek_v4
