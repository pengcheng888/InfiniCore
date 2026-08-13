#include "infinicore/ops/deepseek_v4_shared_experts_impl_int8_marlin.hpp"

#include "../deepseek_v4_fused_experts_impl_int8_marlin/deepseek_v4_fused_experts_impl_int8_marlin_kernel.hpp"
#include "deepseek_v4_shared_experts_impl_int8_marlin_kernel.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/ops/deepseek_v4_lightop_moe_marlin.hpp"

#include "../../utils.hpp"

#include <stdexcept>
#include <string>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4SharedExpertsImplInt8Marlin);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4SharedExpertsImplInt8MarlinWorkspace);

namespace {

constexpr int kBlockSize = 16;
constexpr int kSharedExpertTopK = 1;

void guard_device(const Tensor &tensor, const char *op_name) {
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

struct SharedExpertsShape {
    size_t tokens{0};
    size_t hidden{0};
    size_t intermediate{0};
    size_t gate_up{0};
    size_t flat_topk{0};
    size_t padded_tokens{0};
};

SharedExpertsShape infer_shape(const Tensor &hidden_states, const Tensor &w1) {
    SharedExpertsShape shape;
    shape.tokens = hidden_states->size(0);
    shape.hidden = hidden_states->size(1);
    shape.gate_up = w1->size(2) / 64;
    shape.intermediate = shape.gate_up / 2;
    shape.flat_topk = shape.tokens * static_cast<size_t>(kSharedExpertTopK);
    shape.padded_tokens = ((shape.flat_topk + static_cast<size_t>(kBlockSize - 1) + static_cast<size_t>(kBlockSize - 1)) / static_cast<size_t>(kBlockSize)) * static_cast<size_t>(kBlockSize);
    return shape;
}

void select_gemm_modes(const SharedExpertsShape &shape, int &gemm1_mode, int &gemm2_mode) {
    if (gemm1_mode >= 0 && gemm2_mode >= 0) {
        return;
    }
    if (shape.hidden != 4096 || shape.intermediate != 256) {
        throw std::runtime_error("deepseek_v4_shared_experts_impl_int8_marlin_ automatic mode selection currently supports hidden=4096 and local intermediate=256.");
    }
    int selected_gemm1 = 37;
    int selected_gemm2 = 54;
    if (shape.tokens <= 1) {
        selected_gemm1 = 58;
        selected_gemm2 = 16;
    } else if (shape.tokens <= 7) {
        selected_gemm1 = 29;
        selected_gemm2 = 9;
    } else if (shape.tokens <= 16) {
        selected_gemm1 = 29;
        selected_gemm2 = 4;
    } else if (shape.tokens <= 75) {
        selected_gemm1 = 29;
        selected_gemm2 = 12;
    }
    if (gemm1_mode < 0) {
        gemm1_mode = selected_gemm1;
    }
    if (gemm2_mode < 0) {
        gemm2_mode = selected_gemm2;
    }
}

void check_tensors(Tensor output,
                   const Tensor &hidden_states,
                   const Tensor &w1,
                   const Tensor &w2,
                   const Tensor &w1_scale,
                   const Tensor &w2_scale) {
    if (hidden_states->ndim() != 2 || output->shape() != hidden_states->shape()) {
        throw std::runtime_error("deepseek_v4_shared_experts_impl_int8_marlin_ expects output/hidden_states [tokens, hidden] with identical shape.");
    }
    if (hidden_states->dtype() != DataType::BF16 || output->dtype() != DataType::BF16) {
        throw std::runtime_error("deepseek_v4_shared_experts_impl_int8_marlin_ currently expects BF16 output/hidden_states.");
    }
    if (!hidden_states->is_contiguous() || !output->is_contiguous()) {
        throw std::runtime_error("deepseek_v4_shared_experts_impl_int8_marlin_ expects contiguous output/hidden_states.");
    }
    if (w1->ndim() != 3 || w1->size(0) != 1 || w1->size(1) * 64 != hidden_states->size(1) || w1->size(2) % 128 != 0) {
        throw std::runtime_error("deepseek_v4_shared_experts_impl_int8_marlin_ expects w1 Marlin layout [1, hidden/64, 2*intermediate*64].");
    }
    const auto shape = infer_shape(hidden_states, w1);
    if (w2->ndim() != 3 || w2->size(0) != 1 || w2->size(1) * 64 != shape.intermediate || w2->size(2) != shape.hidden * 64) {
        throw std::runtime_error("deepseek_v4_shared_experts_impl_int8_marlin_ expects w2 Marlin layout [1, intermediate/64, hidden*64].");
    }
    if (w1_scale->ndim() != 3 || w1_scale->size(0) != 1 || w1_scale->size(1) != shape.gate_up || w1_scale->size(2) != 1) {
        throw std::runtime_error("deepseek_v4_shared_experts_impl_int8_marlin_ expects w1_scale [1, 2*intermediate, 1].");
    }
    if (w2_scale->ndim() != 3 || w2_scale->size(0) != 1 || w2_scale->size(1) != shape.hidden || w2_scale->size(2) != 1) {
        throw std::runtime_error("deepseek_v4_shared_experts_impl_int8_marlin_ expects w2_scale [1, hidden, 1].");
    }
}

void check_workspace_tensors(const SharedExpertsShape &shape,
                             Tensor sorted_token_ids,
                             Tensor expert_ids,
                             Tensor num_tokens_post_pad,
                             Tensor topk_weights,
                             Tensor q_hidden,
                             Tensor hidden_scale,
                             Tensor gate_up,
                             Tensor q_activated,
                             Tensor activated_scale,
                             const Tensor &hidden_states) {
    if (sorted_token_ids->shape() != Shape({shape.padded_tokens}) ||
        expert_ids->shape() != Shape({shape.padded_tokens / static_cast<size_t>(kBlockSize)}) ||
        num_tokens_post_pad->shape() != Shape({1}) ||
        topk_weights->shape() != Shape({shape.tokens, static_cast<size_t>(kSharedExpertTopK)})) {
        throw std::runtime_error("deepseek_v4_shared_experts_impl_int8_marlin_ unexpected metadata workspace shape.");
    }
    if (sorted_token_ids->dtype() != DataType::I32 || expert_ids->dtype() != DataType::I32 ||
        num_tokens_post_pad->dtype() != DataType::I32 || topk_weights->dtype() != DataType::F32) {
        throw std::runtime_error("deepseek_v4_shared_experts_impl_int8_marlin_ unexpected metadata workspace dtype.");
    }
    if (q_hidden->shape() != hidden_states->shape() ||
        hidden_scale->shape() != Shape({shape.tokens, 1}) ||
        gate_up->shape() != Shape({shape.tokens, static_cast<size_t>(kSharedExpertTopK), shape.gate_up}) ||
        q_activated->shape() != Shape({shape.flat_topk, shape.intermediate}) ||
        activated_scale->shape() != Shape({shape.flat_topk, 1})) {
        throw std::runtime_error("deepseek_v4_shared_experts_impl_int8_marlin_ unexpected compute workspace shape.");
    }
    if (q_hidden->dtype() != DataType::I8 || hidden_scale->dtype() != DataType::F32 ||
        gate_up->dtype() != hidden_states->dtype() || q_activated->dtype() != DataType::I8 ||
        activated_scale->dtype() != DataType::F32) {
        throw std::runtime_error("deepseek_v4_shared_experts_impl_int8_marlin_ unexpected compute workspace dtype.");
    }
    if (!sorted_token_ids->is_contiguous() || !expert_ids->is_contiguous() || !num_tokens_post_pad->is_contiguous() ||
        !topk_weights->is_contiguous() || !q_hidden->is_contiguous() || !hidden_scale->is_contiguous() ||
        !gate_up->is_contiguous() || !q_activated->is_contiguous() || !activated_scale->is_contiguous()) {
        throw std::runtime_error("deepseek_v4_shared_experts_impl_int8_marlin_ expects contiguous workspace tensors.");
    }
}

void quant_bf16_to_int8_(Tensor output, Tensor scale, const Tensor &input, const char *op_name) {
    if (input->dtype() != DataType::BF16 || output->dtype() != DataType::I8 || scale->dtype() != DataType::F32) {
        throw std::runtime_error(std::string(op_name) + " unexpected quant dtype.");
    }
    if (!input->is_contiguous() || !output->is_contiguous() || !scale->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous quant tensors.");
    }
    const int64_t cols = static_cast<int64_t>(input->size(input->ndim() - 1));
    const int64_t rows = static_cast<int64_t>(input->numel() / static_cast<size_t>(cols));
    deepseek_v4_fused_experts_impl_int8_marlin::launch_per_token_quant_int8_bf16(
        output->data(),
        reinterpret_cast<float *>(scale->data()),
        input->data(),
        rows,
        cols,
        context::getStream());
}

void fill_single_expert_metadata_(Tensor sorted_token_ids,
                                  Tensor expert_ids,
                                  Tensor num_tokens_post_pad,
                                  Tensor topk_weights,
                                  size_t tokens) {
    deepseek_v4_shared_experts_impl_int8_marlin::launch_fill_single_expert_metadata(
        sorted_token_ids->data(),
        expert_ids->data(),
        num_tokens_post_pad->data(),
        topk_weights->data(),
        static_cast<int64_t>(tokens),
        kSharedExpertTopK,
        kBlockSize,
        context::getStream());
}

struct SharedExpertsWorkspace {
    Tensor sorted_token_ids;
    Tensor expert_ids;
    Tensor num_tokens_post_pad;
    Tensor topk_weights;
    Tensor q_hidden;
    Tensor hidden_scale;
    Tensor gate_up;
    Tensor q_activated;
    Tensor activated_scale;

    SharedExpertsWorkspace(Tensor sorted_token_ids_,
                           Tensor expert_ids_,
                           Tensor num_tokens_post_pad_,
                           Tensor topk_weights_,
                           Tensor q_hidden_,
                           Tensor hidden_scale_,
                           Tensor gate_up_,
                           Tensor q_activated_,
                           Tensor activated_scale_)
        : sorted_token_ids(graph::GraphTensor(sorted_token_ids_)),
          expert_ids(graph::GraphTensor(expert_ids_)),
          num_tokens_post_pad(graph::GraphTensor(num_tokens_post_pad_)),
          topk_weights(graph::GraphTensor(topk_weights_)),
          q_hidden(graph::GraphTensor(q_hidden_)),
          hidden_scale(graph::GraphTensor(hidden_scale_)),
          gate_up(graph::GraphTensor(gate_up_)),
          q_activated(graph::GraphTensor(q_activated_)),
          activated_scale(graph::GraphTensor(activated_scale_)) {
    }
};

SharedExpertsWorkspace make_workspace(const Tensor &hidden_states, const Tensor &w1) {
    const auto shape = infer_shape(hidden_states, w1);
    return SharedExpertsWorkspace(
        Tensor::empty({shape.padded_tokens}, DataType::I32, hidden_states->device()),
        Tensor::empty({shape.padded_tokens / static_cast<size_t>(kBlockSize)}, DataType::I32, hidden_states->device()),
        Tensor::empty({1}, DataType::I32, hidden_states->device()),
        Tensor::empty({shape.tokens, static_cast<size_t>(kSharedExpertTopK)}, DataType::F32, hidden_states->device()),
        Tensor::empty(hidden_states->shape(), DataType::I8, hidden_states->device()),
        Tensor::empty({shape.tokens, 1}, DataType::F32, hidden_states->device()),
        Tensor::empty({shape.tokens, static_cast<size_t>(kSharedExpertTopK), shape.gate_up}, hidden_states->dtype(), hidden_states->device()),
        Tensor::empty({shape.flat_topk, shape.intermediate}, DataType::I8, hidden_states->device()),
        Tensor::empty({shape.flat_topk, 1}, DataType::F32, hidden_states->device()));
}

void shared_experts_impl_(Tensor output,
                          const Tensor &hidden_states,
                          const Tensor &w1,
                          const Tensor &w2,
                          const Tensor &w1_scale,
                          const Tensor &w2_scale,
                          int gemm1_mode,
                          int gemm2_mode,
                          int delta,
                          const SharedExpertsWorkspace *workspace,
                          bool metadata_ready) {
    guard_device(output, "deepseek_v4_shared_experts_impl_int8_marlin_");
    guard_device(hidden_states, "deepseek_v4_shared_experts_impl_int8_marlin_");
    guard_device(w1, "deepseek_v4_shared_experts_impl_int8_marlin_");
    guard_device(w2, "deepseek_v4_shared_experts_impl_int8_marlin_");
    check_tensors(output, hidden_states, w1, w2, w1_scale, w2_scale);

    const auto shape = infer_shape(hidden_states, w1);
    select_gemm_modes(shape, gemm1_mode, gemm2_mode);
    auto sorted_token_ids = workspace ? workspace->sorted_token_ids : Tensor::empty({shape.padded_tokens}, DataType::I32, hidden_states->device());
    auto expert_ids = workspace ? workspace->expert_ids : Tensor::empty({shape.padded_tokens / static_cast<size_t>(kBlockSize)}, DataType::I32, hidden_states->device());
    auto num_tokens_post_pad = workspace ? workspace->num_tokens_post_pad : Tensor::empty({1}, DataType::I32, hidden_states->device());
    auto topk_weights = workspace ? workspace->topk_weights : Tensor::empty({shape.tokens, static_cast<size_t>(kSharedExpertTopK)}, DataType::F32, hidden_states->device());
    if (!metadata_ready) {
        fill_single_expert_metadata_(sorted_token_ids, expert_ids, num_tokens_post_pad, topk_weights, shape.tokens);
    }

    auto q_hidden = workspace ? workspace->q_hidden : Tensor::empty(hidden_states->shape(), DataType::I8, hidden_states->device());
    auto hidden_scale = workspace ? workspace->hidden_scale : Tensor::empty({shape.tokens, 1}, DataType::F32, hidden_states->device());
    quant_bf16_to_int8_(q_hidden, hidden_scale, hidden_states, "deepseek_v4_shared_experts_impl_int8_marlin_");

    auto gate_up = workspace ? workspace->gate_up : Tensor::empty({shape.tokens, static_cast<size_t>(kSharedExpertTopK), shape.gate_up}, hidden_states->dtype(), hidden_states->device());
    deepseek_v4_lightop_moe_gemm_marlin_w8a8_(
        q_hidden,
        w1,
        gate_up,
        hidden_scale,
        w1_scale,
        std::nullopt,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        kSharedExpertTopK,
        gemm1_mode,
        delta);

    auto q_activated = workspace ? workspace->q_activated : Tensor::empty({shape.flat_topk, shape.intermediate}, DataType::I8, hidden_states->device());
    auto activated_scale = workspace ? workspace->activated_scale : Tensor::empty({shape.flat_topk, 1}, DataType::F32, hidden_states->device());
    deepseek_v4_lightop_fuse_silu_mul_quant_(
        q_activated,
        activated_scale,
        gate_up->view({shape.flat_topk, shape.gate_up}),
        std::nullopt,
        1,
        -1,
        std::nullopt);

    deepseek_v4_lightop_moe_gemm_marlin_w8a8_(
        q_activated,
        w2,
        output,
        activated_scale,
        w2_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        1,
        gemm2_mode,
        delta);
}

struct SharedExpertsGraphMeta {
    graph::GraphTensor output;
    graph::GraphTensor hidden_states;
    graph::GraphTensor w1;
    graph::GraphTensor w2;
    graph::GraphTensor w1_scale;
    graph::GraphTensor w2_scale;
    SharedExpertsWorkspace workspace;
    int gemm1_mode;
    int gemm2_mode;
    int delta;

    SharedExpertsGraphMeta(Tensor output_,
                           const Tensor &hidden_states_,
                           const Tensor &w1_,
                           const Tensor &w2_,
                           const Tensor &w1_scale_,
                           const Tensor &w2_scale_,
                           int gemm1_mode_,
                           int gemm2_mode_,
                           int delta_)
        : output(output_),
          hidden_states(hidden_states_),
          w1(w1_),
          w2(w2_),
          w1_scale(w1_scale_),
          w2_scale(w2_scale_),
          workspace(make_workspace(hidden_states_, w1_)),
          gemm1_mode(gemm1_mode_),
          gemm2_mode(gemm2_mode_),
          delta(delta_) {
    }
};

void *plan_shared_experts(Tensor output,
                          const Tensor &hidden_states,
                          const Tensor &w1,
                          const Tensor &w2,
                          const Tensor &w1_scale,
                          const Tensor &w2_scale,
                          int gemm1_mode,
                          int gemm2_mode,
                          int delta) {
    check_tensors(output, hidden_states, w1, w2, w1_scale, w2_scale);
    return new SharedExpertsGraphMeta(output, hidden_states, w1, w2, w1_scale, w2_scale, gemm1_mode, gemm2_mode, delta);
}

void run_shared_experts(void *meta) {
    auto *m = static_cast<SharedExpertsGraphMeta *>(meta);
    shared_experts_impl_(
        m->output,
        m->hidden_states,
        m->w1,
        m->w2,
        m->w1_scale,
        m->w2_scale,
        m->gemm1_mode,
        m->gemm2_mode,
        m->delta,
        &m->workspace,
        false);
}

void cleanup_shared_experts(void **meta) {
    if (meta && *meta) {
        delete static_cast<SharedExpertsGraphMeta *>(*meta);
        *meta = nullptr;
    }
}

struct SharedExpertsWorkspaceGraphMeta {
    graph::GraphTensor output;
    graph::GraphTensor hidden_states;
    graph::GraphTensor w1;
    graph::GraphTensor w2;
    graph::GraphTensor w1_scale;
    graph::GraphTensor w2_scale;
    SharedExpertsWorkspace workspace;
    int gemm1_mode;
    int gemm2_mode;
    int delta;

    SharedExpertsWorkspaceGraphMeta(Tensor output_,
                                    const Tensor &hidden_states_,
                                    const Tensor &w1_,
                                    const Tensor &w2_,
                                    const Tensor &w1_scale_,
                                    const Tensor &w2_scale_,
                                    Tensor sorted_token_ids_,
                                    Tensor expert_ids_,
                                    Tensor num_tokens_post_pad_,
                                    Tensor topk_weights_,
                                    Tensor q_hidden_,
                                    Tensor hidden_scale_,
                                    Tensor gate_up_,
                                    Tensor q_activated_,
                                    Tensor activated_scale_,
                                    int gemm1_mode_,
                                    int gemm2_mode_,
                                    int delta_)
        : output(output_),
          hidden_states(hidden_states_),
          w1(w1_),
          w2(w2_),
          w1_scale(w1_scale_),
          w2_scale(w2_scale_),
          workspace(sorted_token_ids_,
                    expert_ids_,
                    num_tokens_post_pad_,
                    topk_weights_,
                    q_hidden_,
                    hidden_scale_,
                    gate_up_,
                    q_activated_,
                    activated_scale_),
          gemm1_mode(gemm1_mode_),
          gemm2_mode(gemm2_mode_),
          delta(delta_) {
    }
};

void *plan_shared_experts_workspace(Tensor output,
                                    const Tensor &hidden_states,
                                    const Tensor &w1,
                                    const Tensor &w2,
                                    const Tensor &w1_scale,
                                    const Tensor &w2_scale,
                                    Tensor sorted_token_ids,
                                    Tensor expert_ids,
                                    Tensor num_tokens_post_pad,
                                    Tensor topk_weights,
                                    Tensor q_hidden,
                                    Tensor hidden_scale,
                                    Tensor gate_up,
                                    Tensor q_activated,
                                    Tensor activated_scale,
                                    int gemm1_mode,
                                    int gemm2_mode,
                                    int delta) {
    check_tensors(output, hidden_states, w1, w2, w1_scale, w2_scale);
    const auto shape = infer_shape(hidden_states, w1);
    check_workspace_tensors(shape,
                            sorted_token_ids,
                            expert_ids,
                            num_tokens_post_pad,
                            topk_weights,
                            q_hidden,
                            hidden_scale,
                            gate_up,
                            q_activated,
                            activated_scale,
                            hidden_states);
    return new SharedExpertsWorkspaceGraphMeta(output,
                                               hidden_states,
                                               w1,
                                               w2,
                                               w1_scale,
                                               w2_scale,
                                               sorted_token_ids,
                                               expert_ids,
                                               num_tokens_post_pad,
                                               topk_weights,
                                               q_hidden,
                                               hidden_scale,
                                               gate_up,
                                               q_activated,
                                               activated_scale,
                                               gemm1_mode,
                                               gemm2_mode,
                                               delta);
}

void run_shared_experts_workspace(void *meta) {
    auto *m = static_cast<SharedExpertsWorkspaceGraphMeta *>(meta);
    shared_experts_impl_(
        m->output,
        m->hidden_states,
        m->w1,
        m->w2,
        m->w1_scale,
        m->w2_scale,
        m->gemm1_mode,
        m->gemm2_mode,
        m->delta,
        &m->workspace,
        true);
}

void cleanup_shared_experts_workspace(void **meta) {
    if (meta && *meta) {
        delete static_cast<SharedExpertsWorkspaceGraphMeta *>(*meta);
        *meta = nullptr;
    }
}

} // namespace

DeepseekV4SharedExpertsImplInt8Marlin::DeepseekV4SharedExpertsImplInt8Marlin(Tensor output,
                                                                             const Tensor &hidden_states,
                                                                             const Tensor &w1,
                                                                             const Tensor &w2,
                                                                             const Tensor &w1_scale,
                                                                             const Tensor &w2_scale,
                                                                             int gemm1_mode,
                                                                             int gemm2_mode,
                                                                             int delta) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, hidden_states, w1, w2, w1_scale, w2_scale);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output, hidden_states, w1, w2, w1_scale, w2_scale, gemm1_mode, gemm2_mode, delta);
}

void DeepseekV4SharedExpertsImplInt8Marlin::execute(Tensor output,
                                                    const Tensor &hidden_states,
                                                    const Tensor &w1,
                                                    const Tensor &w2,
                                                    const Tensor &w1_scale,
                                                    const Tensor &w2_scale,
                                                    int gemm1_mode,
                                                    int gemm2_mode,
                                                    int delta) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4SharedExpertsImplInt8Marlin, output, hidden_states, w1, w2, w1_scale, w2_scale, gemm1_mode, gemm2_mode, delta);
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4SharedExpertsImplInt8Marlin,
                                       &plan_shared_experts,
                                       &run_shared_experts,
                                       &cleanup_shared_experts);

DeepseekV4SharedExpertsImplInt8MarlinWorkspace::DeepseekV4SharedExpertsImplInt8MarlinWorkspace(Tensor output,
                                                                                               const Tensor &hidden_states,
                                                                                               const Tensor &w1,
                                                                                               const Tensor &w2,
                                                                                               const Tensor &w1_scale,
                                                                                               const Tensor &w2_scale,
                                                                                               Tensor sorted_token_ids,
                                                                                               Tensor expert_ids,
                                                                                               Tensor num_tokens_post_pad,
                                                                                               Tensor topk_weights,
                                                                                               Tensor q_hidden,
                                                                                               Tensor hidden_scale,
                                                                                               Tensor gate_up,
                                                                                               Tensor q_activated,
                                                                                               Tensor activated_scale,
                                                                                               int gemm1_mode,
                                                                                               int gemm2_mode,
                                                                                               int delta) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output,
                                          hidden_states,
                                          w1,
                                          w2,
                                          w1_scale,
                                          w2_scale,
                                          sorted_token_ids,
                                          expert_ids,
                                          num_tokens_post_pad,
                                          topk_weights,
                                          q_hidden,
                                          hidden_scale,
                                          gate_up,
                                          q_activated,
                                          activated_scale);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(),
                                 output,
                                 hidden_states,
                                 w1,
                                 w2,
                                 w1_scale,
                                 w2_scale,
                                 sorted_token_ids,
                                 expert_ids,
                                 num_tokens_post_pad,
                                 topk_weights,
                                 q_hidden,
                                 hidden_scale,
                                 gate_up,
                                 q_activated,
                                 activated_scale,
                                 gemm1_mode,
                                 gemm2_mode,
                                 delta);
}

void DeepseekV4SharedExpertsImplInt8MarlinWorkspace::execute(Tensor output,
                                                             const Tensor &hidden_states,
                                                             const Tensor &w1,
                                                             const Tensor &w2,
                                                             const Tensor &w1_scale,
                                                             const Tensor &w2_scale,
                                                             Tensor sorted_token_ids,
                                                             Tensor expert_ids,
                                                             Tensor num_tokens_post_pad,
                                                             Tensor topk_weights,
                                                             Tensor q_hidden,
                                                             Tensor hidden_scale,
                                                             Tensor gate_up,
                                                             Tensor q_activated,
                                                             Tensor activated_scale,
                                                             int gemm1_mode,
                                                             int gemm2_mode,
                                                             int delta) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4SharedExpertsImplInt8MarlinWorkspace,
                                      output,
                                      hidden_states,
                                      w1,
                                      w2,
                                      w1_scale,
                                      w2_scale,
                                      sorted_token_ids,
                                      expert_ids,
                                      num_tokens_post_pad,
                                      topk_weights,
                                      q_hidden,
                                      hidden_scale,
                                      gate_up,
                                      q_activated,
                                      activated_scale,
                                      gemm1_mode,
                                      gemm2_mode,
                                      delta);
}

static bool registered_shared_experts_workspace = []() {
    DeepseekV4SharedExpertsImplInt8MarlinWorkspace::plan_dispatcher().registerAll(&plan_shared_experts_workspace, false);
    DeepseekV4SharedExpertsImplInt8MarlinWorkspace::run_dispatcher().registerAll(&run_shared_experts_workspace, false);
    DeepseekV4SharedExpertsImplInt8MarlinWorkspace::cleanup_dispatcher().registerAll(&cleanup_shared_experts_workspace, false);
    return true;
}();

void deepseek_v4_shared_experts_impl_int8_marlin_(Tensor output,
                                                  const Tensor &hidden_states,
                                                  const Tensor &w1,
                                                  const Tensor &w2,
                                                  const Tensor &w1_scale,
                                                  const Tensor &w2_scale,
                                                  int gemm1_mode,
                                                  int gemm2_mode,
                                                  int delta) {
    DeepseekV4SharedExpertsImplInt8Marlin::execute(output, hidden_states, w1, w2, w1_scale, w2_scale, gemm1_mode, gemm2_mode, delta);
}

void deepseek_v4_shared_experts_impl_int8_marlin_prepare_metadata_(Tensor sorted_token_ids,
                                                                   Tensor expert_ids,
                                                                   Tensor num_tokens_post_pad,
                                                                   Tensor topk_weights,
                                                                   size_t tokens) {
    guard_device(sorted_token_ids, "deepseek_v4_shared_experts_impl_int8_marlin_prepare_metadata_");
    if (topk_weights->ndim() != 2 || topk_weights->size(0) != tokens || topk_weights->size(1) != static_cast<size_t>(kSharedExpertTopK)) {
        throw std::runtime_error("deepseek_v4_shared_experts_impl_int8_marlin_prepare_metadata_ expects topk_weights [tokens, 1].");
    }
    const size_t flat_topk = tokens * static_cast<size_t>(kSharedExpertTopK);
    const size_t padded_tokens = ((flat_topk + static_cast<size_t>(kBlockSize - 1) + static_cast<size_t>(kBlockSize - 1)) / static_cast<size_t>(kBlockSize)) * static_cast<size_t>(kBlockSize);
    if (sorted_token_ids->shape() != Shape({padded_tokens}) ||
        expert_ids->shape() != Shape({padded_tokens / static_cast<size_t>(kBlockSize)}) ||
        num_tokens_post_pad->shape() != Shape({1})) {
        throw std::runtime_error("deepseek_v4_shared_experts_impl_int8_marlin_prepare_metadata_ unexpected metadata tensor shape.");
    }
    if (sorted_token_ids->dtype() != DataType::I32 || expert_ids->dtype() != DataType::I32 ||
        num_tokens_post_pad->dtype() != DataType::I32 || topk_weights->dtype() != DataType::F32) {
        throw std::runtime_error("deepseek_v4_shared_experts_impl_int8_marlin_prepare_metadata_ unexpected metadata tensor dtype.");
    }
    if (!sorted_token_ids->is_contiguous() || !expert_ids->is_contiguous() ||
        !num_tokens_post_pad->is_contiguous() || !topk_weights->is_contiguous()) {
        throw std::runtime_error("deepseek_v4_shared_experts_impl_int8_marlin_prepare_metadata_ expects contiguous metadata tensors.");
    }
    fill_single_expert_metadata_(sorted_token_ids, expert_ids, num_tokens_post_pad, topk_weights, tokens);
}

void deepseek_v4_shared_experts_impl_int8_marlin_workspace_(Tensor output,
                                                            const Tensor &hidden_states,
                                                            const Tensor &w1,
                                                            const Tensor &w2,
                                                            const Tensor &w1_scale,
                                                            const Tensor &w2_scale,
                                                            Tensor sorted_token_ids,
                                                            Tensor expert_ids,
                                                            Tensor num_tokens_post_pad,
                                                            Tensor topk_weights,
                                                            Tensor q_hidden,
                                                            Tensor hidden_scale,
                                                            Tensor gate_up,
                                                            Tensor q_activated,
                                                            Tensor activated_scale,
                                                            int gemm1_mode,
                                                            int gemm2_mode,
                                                            int delta) {
    DeepseekV4SharedExpertsImplInt8MarlinWorkspace::execute(output,
                                                            hidden_states,
                                                            w1,
                                                            w2,
                                                            w1_scale,
                                                            w2_scale,
                                                            sorted_token_ids,
                                                            expert_ids,
                                                            num_tokens_post_pad,
                                                            topk_weights,
                                                            q_hidden,
                                                            hidden_scale,
                                                            gate_up,
                                                            q_activated,
                                                            activated_scale,
                                                            gemm1_mode,
                                                            gemm2_mode,
                                                            delta);
}

} // namespace infinicore::op
