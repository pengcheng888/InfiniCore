#include "infinicore/ops/deepseek_v4_moe_w8a8.hpp"

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

} // namespace

void deepseek_v4_moe_w8a8_naive_(Tensor y,
                                     const Tensor &x,
                                     const Tensor &topk_weights,
                                     const Tensor &topk_indices,
                                     const Tensor &w13,
                                     const Tensor &w13_scale,
                                     const Tensor &w2,
                                     const Tensor &w2_scale,
                                     double swiglu_limit) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_accelerator_tensor(x, "deepseek_v4_moe_w8a8_naive_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    if (x->ndim() != 2 || topk_weights->ndim() != 2 || topk_indices->ndim() != 2 ||
        w13->ndim() != 3 || w13_scale->ndim() != 3 || w2->ndim() != 3 || w2_scale->ndim() != 3 ||
        y->shape() != x->shape()) {
        throw std::runtime_error("deepseek_v4_moe_w8a8_naive_ shape/rank mismatch.");
    }
    if (topk_weights->shape() != topk_indices->shape()) {
        throw std::runtime_error("deepseek_v4_moe_w8a8_naive_ topk shape mismatch.");
    }
    if (w13->size(0) != w2->size(0) || w13->size(1) != w2->size(2) * 2 ||
        w13->size(2) != x->size(1) || w2->size(1) != x->size(1)) {
        throw std::runtime_error("deepseek_v4_moe_w8a8_naive_ packed weight shape mismatch.");
    }

    auto y_at = infinicore::adaptor::to_aten_tensor(y);
    auto x_at = infinicore::adaptor::to_aten_tensor(x);
    auto weights_at = infinicore::adaptor::to_aten_tensor(topk_weights).to(at::kFloat);
    auto indices_at = infinicore::adaptor::to_aten_tensor(topk_indices).to(at::kLong);
    auto w13_at = infinicore::adaptor::to_aten_tensor(w13);
    auto w13_scale_at = infinicore::adaptor::to_aten_tensor(w13_scale);
    auto w2_at = infinicore::adaptor::to_aten_tensor(w2);
    auto w2_scale_at = infinicore::adaptor::to_aten_tensor(w2_scale);

    const int64_t tokens = x_at.size(0);
    const int64_t hidden = x_at.size(1);
    const int64_t topk = indices_at.size(1);
    const int64_t num_experts = w13_at.size(0);
    const int64_t intermediate = w2_at.size(2);

    auto out = at::zeros({tokens, hidden}, x_at.options().dtype(at::kFloat));
    auto token_arange = at::arange(tokens, indices_at.options()).repeat_interleave(topk);
    auto flat_ids = indices_at.reshape({tokens * topk});
    auto flat_weights = weights_at.reshape({tokens * topk});
    auto x_float = x_at.to(at::kFloat);

    for (int64_t expert = 0; expert < num_experts; ++expert) {
        auto route_pos = at::nonzero(flat_ids == expert).flatten();
        if (route_pos.numel() == 0) {
            continue;
        }
        auto token_idx = token_arange.index_select(0, route_pos);
        auto x_e = x_float.index_select(0, token_idx);
        auto route_weight = flat_weights.index_select(0, route_pos).unsqueeze(1);

        auto w13_e = w13_at[expert].to(at::kFloat) * w13_scale_at[expert].to(at::kFloat);
        auto gate_up = at::matmul(x_e, w13_e.transpose(0, 1));
        auto gate = gate_up.slice(1, 0, intermediate);
        auto up = gate_up.slice(1, intermediate, 2 * intermediate);
        gate = at::minimum(gate, at::full({}, swiglu_limit, gate.options()));
        up = at::clamp(up, -swiglu_limit, swiglu_limit);
        auto act = (gate / (1.0 + at::exp(-gate))) * up;

        auto w2_e = w2_at[expert].to(at::kFloat) * w2_scale_at[expert].to(at::kFloat);
        auto down = at::matmul(act, w2_e.transpose(0, 1)) * route_weight;
        out.index_add_(0, token_idx, down);
    }

    y_at.copy_(out.to(y_at.scalar_type()));
#else
    (void)y;
    (void)x;
    (void)topk_weights;
    (void)topk_indices;
    (void)w13;
    (void)w13_scale;
    (void)w2;
    (void)w2_scale;
    (void)swiglu_limit;
    throw std::runtime_error("deepseek_v4_moe_w8a8_naive_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}


} // namespace infinicore::op
