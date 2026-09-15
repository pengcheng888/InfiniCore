#include "infinicore/ops/deepseek_v4/moe_w8a8.hpp"

#include "../../aten_utils.hpp"

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

#include <set>
#include <stdexcept>
#include <string>

namespace infinicore::op::deepseek_v4 {
namespace {

constexpr int64_t kMaxSelectedRoutes = 64;

void check_accelerator_tensor_v2(const Tensor &tensor, const char *op_name) {
    detail::check_build_device(tensor, op_name);
}

bool should_use_selected_routes(const Tensor &x, const Tensor &topk_indices) {
    if (x->ndim() != 2 || topk_indices->ndim() != 2 ||
        x->size(0) != topk_indices->size(0)) {
        return false;
    }
    const auto tokens = static_cast<int64_t>(x->size(0));
    const auto topk = static_cast<int64_t>(topk_indices->size(1));
    return tokens <= kMaxSelectedRoutes &&
           topk <= kMaxSelectedRoutes &&
           tokens * topk <= kMaxSelectedRoutes;
}

#if defined(ENABLE_ATEN)
std::set<int64_t> collect_active_experts(const at::Tensor &flat_ids, int64_t num_experts) {
    auto flat_ids_cpu = flat_ids.to(at::kCPU);
    const auto *ids = flat_ids_cpu.data_ptr<int64_t>();
    std::set<int64_t> active_experts;
    for (int64_t route = 0; route < flat_ids_cpu.numel(); ++route) {
        const int64_t expert = ids[route];
        if (expert >= 0 && expert < num_experts) {
            active_experts.insert(expert);
        }
    }
    return active_experts;
}
#endif

} // namespace

void moe_w8a8_v2_aten_(Tensor y,
                       const Tensor &x,
                       const Tensor &topk_weights,
                       const Tensor &topk_indices,
                       const Tensor &w13,
                       const Tensor &w13_scale,
                       const Tensor &w2,
                       const Tensor &w2_scale,
                       double swiglu_limit) {
#if defined(ENABLE_ATEN) && defined(INFINICORE_DSV4_ACCELERATOR_API)
    if (!should_use_selected_routes(x, topk_indices)) {
        moe_w8a8_aten_(y,
                       x,
                       topk_weights,
                       topk_indices,
                       w13,
                       w13_scale,
                       w2,
                       w2_scale,
                       swiglu_limit);
        return;
    }

    check_accelerator_tensor_v2(x, "deepseek_v4::moe_w8a8_v2_aten_");
    detail::prepare_aten_call(x, "deepseek_v4::moe_w8a8_v2_aten_");

    if (topk_weights->ndim() != 2 || w13->ndim() != 3 ||
        w13_scale->ndim() != 3 || w2->ndim() != 3 ||
        w2_scale->ndim() != 3 || y->shape() != x->shape()) {
        throw std::runtime_error("deepseek_v4::moe_w8a8_v2_aten_ shape/rank mismatch.");
    }
    if (topk_weights->shape() != topk_indices->shape()) {
        throw std::runtime_error("deepseek_v4::moe_w8a8_v2_aten_ topk shape mismatch.");
    }
    if (w13->size(0) != w2->size(0) || w13->size(1) != w2->size(2) * 2 ||
        w13->size(2) != x->size(1) || w2->size(1) != x->size(1)) {
        throw std::runtime_error("deepseek_v4::moe_w8a8_v2_aten_ packed weight shape mismatch.");
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
    const int64_t routes = tokens * topk;
    const int64_t num_experts = w13_at.size(0);
    const int64_t intermediate = w2_at.size(2);

    auto out = at::zeros({tokens, hidden}, x_at.options().dtype(at::kFloat));
    if (routes == 0) {
        y_at.copy_(out.to(y_at.scalar_type()));
        return;
    }

    auto token_ids = at::arange(tokens, indices_at.options()).repeat_interleave(topk);
    auto flat_ids = indices_at.reshape({routes});
    auto flat_weights = weights_at.reshape({routes});
    auto x_float = x_at.to(at::kFloat);

    // Preserve V1's expert and route ordering while skipping its scan over inactive experts.
    const auto active_experts = collect_active_experts(flat_ids, num_experts);
    for (const int64_t expert : active_experts) {
        auto route_pos = at::nonzero(flat_ids == expert).flatten();
        auto token_idx = token_ids.index_select(0, route_pos);
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
    throw std::runtime_error("deepseek_v4::moe_w8a8_v2_aten_ requires an ATen-enabled HYGON/NVIDIA/METAX/ILUVATAR build.");
#endif
}

} // namespace infinicore::op::deepseek_v4
