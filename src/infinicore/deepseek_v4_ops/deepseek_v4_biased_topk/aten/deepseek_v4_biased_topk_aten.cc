#include "infinicore/ops/deepseek_v4_biased_topk.hpp"

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
#include <tuple>

namespace infinicore::op {
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

void check_shapes(const Tensor &topk_weights,
                  const Tensor &topk_indices,
                  const Tensor &router_logits,
                  const Tensor &correction_bias) {
    if (router_logits->ndim() != 2) {
        throw std::runtime_error("deepseek_v4_topk_aten_ expects router_logits to be 2-D.");
    }
    if (correction_bias->ndim() != 1 || correction_bias->size(0) != router_logits->size(1)) {
        throw std::runtime_error("deepseek_v4_topk_aten_ correction_bias shape mismatch.");
    }
    if (topk_weights->shape() != topk_indices->shape()) {
        throw std::runtime_error("deepseek_v4_topk_aten_ topk weight/index shape mismatch.");
    }
    if (topk_weights->ndim() != 2 || topk_weights->size(0) != router_logits->size(0)) {
        throw std::runtime_error("deepseek_v4_topk_aten_ output shape mismatch.");
    }
}

} // namespace

void deepseek_v4_topk_aten_(Tensor topk_weights,
                             Tensor topk_indices,
                             const Tensor &router_logits,
                             const Tensor &correction_bias,
                             bool renormalize) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_accelerator_tensor(router_logits, "deepseek_v4_topk_aten_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    check_shapes(topk_weights, topk_indices, router_logits, correction_bias);

    auto weights_at = infinicore::adaptor::to_aten_tensor(topk_weights);
    auto indices_at = infinicore::adaptor::to_aten_tensor(topk_indices);
    auto logits_at = infinicore::adaptor::to_aten_tensor(router_logits).to(at::kFloat);
    auto bias_at = infinicore::adaptor::to_aten_tensor(correction_bias).to(at::kFloat);
    const auto k = topk_weights->size(1);

    auto scores = at::sqrt(at::softplus(logits_at));
    auto choice_scores = scores + bias_at.unsqueeze(0);
    auto topk = at::topk(choice_scores, static_cast<int64_t>(k), -1, true, true);
    auto selected = std::get<1>(topk);
    auto gathered = scores.gather(1, selected);
    if (renormalize) {
        gathered = gathered / gathered.sum(-1, true);
    }

    weights_at.copy_(gathered.to(weights_at.scalar_type()));
    indices_at.copy_(selected.to(indices_at.scalar_type()));
#else
    (void)topk_weights;
    (void)topk_indices;
    (void)router_logits;
    (void)correction_bias;
    (void)renormalize;
    throw std::runtime_error("deepseek_v4_topk_aten_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
