#include "infinicore/ops/deepseek_v4_fused_experts_impl_int8_marlin.hpp"

#include "deepseek_v4_fused_experts_impl_int8_marlin_kernel.hpp"
#include "infinicore/device.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops/deepseek_v4_lightop_moe_marlin.hpp"

#include "../../utils.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4FusedExpertsImplInt8Marlin);

namespace {

bool fused_experts_graph_debug_enabled() {
    static const bool enabled = []() {
        const char *value = std::getenv("INFINICORE_GRAPH_DEBUG");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

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

struct MarlinGemmConfig {
    size_t max_tokens{0};
    int block_size_m{16};
    int mode{54};
    int delta{1};
};

struct MarlinConfig {
    MarlinGemmConfig gemm1;
    MarlinGemmConfig gemm2;
    bool supported{false};
};

struct FusedExpertsShape {
    size_t num_tokens{0};
    size_t hidden_size{0};
    size_t top_k{0};
    size_t num_experts{0};
    size_t intermediate_size{0};
    size_t gate_up_size{0};
    size_t flat_topk{0};
    size_t max_num_tokens_padded{0};
    MarlinConfig config;
};

template <size_t N>
MarlinGemmConfig select_marlin_gemm_config(size_t num_tokens, const MarlinGemmConfig (&configs)[N]) {
    for (const auto &config : configs) {
        if (num_tokens <= config.max_tokens) {
            return config;
        }
    }
    return configs[N - 1];
}

MarlinGemmConfig disable_asm_marlin_config(MarlinGemmConfig config, const MarlinGemmConfig &fallback) {
    return config.mode >= 1000 ? fallback : config;
}

MarlinConfig select_deepseek_v4_marlin_config(size_t num_tokens,
                                              size_t hidden_size,
                                              size_t intermediate_size,
                                              size_t top_k,
                                              bool graph_safe_config) {
    MarlinConfig config;
    if (hidden_size == 7168 && intermediate_size == 256 && top_k == 8) {
        config.supported = true;
        if (num_tokens <= 1) {
            config.gemm1.mode = 21;
            config.gemm2.mode = 25;
        } else if (num_tokens <= 7) {
            config.gemm1.mode = 78;
            config.gemm2.mode = 73;
        } else if (num_tokens <= 16) {
            config.gemm1.mode = 29;
            config.gemm2.mode = 12;
        } else if (num_tokens <= 75) {
            config.gemm1.mode = 55;
            config.gemm2.mode = 54;
        }
    } else if (hidden_size == 4096 && intermediate_size == 256 && top_k == 6) {
        // Keep the Hygon gfx936/CU80 LightOp Marlin tuning table in C++ so the
        // hot path does not call Python to query get_moe_cuda_marlin_config.
        static constexpr MarlinGemmConfig gemm1_configs[] = {
            {1, 16, 58, 1},
            {2, 16, 19, 1},
            {4, 16, 58, 1},
            {8, 16, 29, 1},
            {16, 16, 29, 1},
            {32, 16, 29, 1},
            {64, 16, 29, 1},
            {128, 16, 29, 1},
            {256, 16, 37, 1},
            {512, 16, 51, 1},
            {1024, 32, 1002, 1},
            {2048, 32, 1002, 1},
            {4096, 128, 1000, 1},
            {6144, 128, 1000, 1},
            {8192, 128, 1000, 1},
        };
        static constexpr MarlinGemmConfig gemm2_configs[] = {
            {1, 16, 16, 1},
            {2, 16, 76, 1},
            {4, 16, 21, 1},
            {8, 16, 9, 1},
            {16, 16, 4, 1},
            {32, 16, 12, 1},
            {64, 16, 12, 1},
            {128, 16, 55, 1},
            {256, 16, 54, 1},
            {512, 16, 57, 1},
            {1024, 16, 94, 2},
            {2048, 32, 568, 1},
            {4096, 32, 568, 4},
            {6144, 32, 568, 4},
            {8192, 32, 568, 4},
        };
        config.supported = true;
        config.gemm1 = select_marlin_gemm_config(num_tokens, gemm1_configs);
        config.gemm2 = select_marlin_gemm_config(num_tokens, gemm2_configs);
        if (graph_safe_config) {
            config.gemm1 = disable_asm_marlin_config(config.gemm1, {0, 16, 37, 1});
            config.gemm2 = disable_asm_marlin_config(config.gemm2, {0, 16, 54, 1});
        }
    }
    return config;
}

FusedExpertsShape infer_fused_experts_shape(const Tensor &hidden_states,
                                            const Tensor &w1,
                                            const Tensor &topk_ids,
                                            int64_t global_num_experts,
                                            bool graph_safe_config = false) {
    FusedExpertsShape shape;
    shape.num_tokens = hidden_states->size(0);
    shape.hidden_size = hidden_states->size(1);
    shape.top_k = topk_ids->size(1);
    shape.num_experts = static_cast<size_t>(global_num_experts > 0 ? global_num_experts : static_cast<int64_t>(w1->size(0)));
    shape.intermediate_size = w1->size(2) / 128;
    shape.gate_up_size = shape.intermediate_size * 2;
    shape.flat_topk = shape.num_tokens * shape.top_k;
    shape.config = select_deepseek_v4_marlin_config(
        shape.num_tokens,
        shape.hidden_size,
        shape.intermediate_size,
        shape.top_k,
        graph_safe_config);
    if (!shape.config.supported) {
        throw std::runtime_error("deepseek_v4_fused_experts_impl_int8_marlin_ only supports DeepSeek-V4 Marlin shapes hidden=4096/topk=6/local_intermediate=256 or hidden=7168/topk=8/local_intermediate=256.");
    }

    const int block_size = shape.config.gemm1.block_size_m;
    shape.max_num_tokens_padded = shape.flat_topk + shape.num_experts * static_cast<size_t>(block_size - 1);
    shape.max_num_tokens_padded = ((shape.max_num_tokens_padded + static_cast<size_t>(block_size - 1)) / static_cast<size_t>(block_size)) * static_cast<size_t>(block_size);
    if (shape.flat_topk < shape.num_experts) {
        shape.max_num_tokens_padded = std::min(shape.flat_topk * static_cast<size_t>(block_size), shape.max_num_tokens_padded);
    }
    return shape;
}

void check_input_shapes(const Tensor &output,
                        const Tensor &hidden_states,
                        const Tensor &w1,
                        const Tensor &topk_weights,
                        const Tensor &topk_ids,
                        const Tensor &w1_scale,
                        const Tensor &w2_scale,
                        const std::optional<Tensor> &shared_output) {
    if (hidden_states->ndim() != 2 || output->shape() != hidden_states->shape()) {
        throw std::runtime_error("deepseek_v4_fused_experts_impl_int8_marlin_ expects output/hidden [tokens, hidden] with identical shape.");
    }
    if (w1->ndim() != 3 || w1->size(1) * 64 != hidden_states->size(1) || w1->size(2) % 128 != 0) {
        throw std::runtime_error("deepseek_v4_fused_experts_impl_int8_marlin_ expects w1 Marlin layout [experts, hidden/64, 2*intermediate*64].");
    }
    if (topk_weights->shape() != topk_ids->shape() || topk_ids->ndim() != 2 || topk_ids->size(0) != hidden_states->size(0)) {
        throw std::runtime_error("deepseek_v4_fused_experts_impl_int8_marlin_ expects topk_weights/topk_ids [tokens, topk].");
    }
    if (w1_scale->ndim() != 3 || w2_scale->ndim() != 3 || w1_scale->size(0) != w1->size(0) || w2_scale->size(0) != w1->size(0)) {
        throw std::runtime_error("deepseek_v4_fused_experts_impl_int8_marlin_ expects per-expert scale tensors.");
    }
    if (shared_output.has_value() && ((*shared_output)->shape() != hidden_states->shape() || (*shared_output)->dtype() != hidden_states->dtype())) {
        throw std::runtime_error("deepseek_v4_fused_experts_impl_int8_marlin_ expects shared_output [tokens, hidden] with the same dtype as hidden_states.");
    }
}

bool same_storage(const Tensor &a, const Tensor &b) {
    return a && b && a->data() == b->data() && a->shape() == b->shape();
}

void lmslim_per_token_quant_int8_bf16_(Tensor output, Tensor scale, const Tensor &input) {
    if (input->dtype() != DataType::BF16) {
        throw std::runtime_error("deepseek_v4_fused_experts_impl_int8_marlin_ native quant currently expects BF16 input.");
    }
    if (input->ndim() < 2 || !input->is_contiguous() || !output->is_contiguous() || !scale->is_contiguous()) {
        throw std::runtime_error("deepseek_v4_fused_experts_impl_int8_marlin_ native quant expects contiguous tensors with ndim >= 2.");
    }
    const int64_t cols = static_cast<int64_t>(input->size(input->ndim() - 1));
    const int64_t rows = static_cast<int64_t>(input->numel() / static_cast<size_t>(cols));
    if (output->dtype() != DataType::I8 || output->shape() != input->shape() || scale->dtype() != DataType::F32 || scale->numel() != static_cast<size_t>(rows)) {
        throw std::runtime_error("deepseek_v4_fused_experts_impl_int8_marlin_ native quant unexpected output/scale shape.");
    }
    deepseek_v4_fused_experts_impl_int8_marlin::launch_per_token_quant_int8_bf16(
        output->data(),
        reinterpret_cast<float *>(scale->data()),
        input->data(),
        rows,
        cols,
        context::getStream());
}

struct FusedExpertsWorkspace {
    Tensor sorted_token_ids;
    Tensor expert_ids;
    Tensor num_tokens_post_pad;
    Tensor q_hidden;
    Tensor hidden_scale;
    Tensor gate_up;
    Tensor q_activated;
    Tensor activated_scale;
    Tensor down;

    FusedExpertsWorkspace(Tensor sorted_token_ids_,
                          Tensor expert_ids_,
                          Tensor num_tokens_post_pad_,
                          Tensor q_hidden_,
                          Tensor hidden_scale_,
                          Tensor gate_up_,
                          Tensor q_activated_,
                          Tensor activated_scale_,
                          Tensor down_)
        : sorted_token_ids(graph::GraphTensor(sorted_token_ids_)),
          expert_ids(graph::GraphTensor(expert_ids_)),
          num_tokens_post_pad(graph::GraphTensor(num_tokens_post_pad_)),
          q_hidden(graph::GraphTensor(q_hidden_)),
          hidden_scale(graph::GraphTensor(hidden_scale_)),
          gate_up(graph::GraphTensor(gate_up_)),
          q_activated(graph::GraphTensor(q_activated_)),
          activated_scale(graph::GraphTensor(activated_scale_)),
          down(graph::GraphTensor(down_)) {
    }
};

FusedExpertsWorkspace make_workspace(const Tensor &hidden_states,
                                     const Tensor &w1,
                                     const Tensor &topk_ids,
                                     int64_t global_num_experts) {
    const auto shape = infer_fused_experts_shape(hidden_states, w1, topk_ids, global_num_experts, true);
    const int block_size = shape.config.gemm1.block_size_m;

    return FusedExpertsWorkspace(
        Tensor::empty({shape.max_num_tokens_padded}, DataType::I32, hidden_states->device()),
        Tensor::empty({(shape.max_num_tokens_padded + static_cast<size_t>(block_size - 1)) / static_cast<size_t>(block_size)}, DataType::I32, hidden_states->device()),
        Tensor::empty({1}, DataType::I32, hidden_states->device()),
        Tensor::empty(hidden_states->shape(), DataType::I8, hidden_states->device()),
        Tensor::empty({shape.num_tokens, 1}, DataType::F32, hidden_states->device()),
        Tensor::empty({shape.num_tokens, shape.top_k, shape.gate_up_size}, hidden_states->dtype(), hidden_states->device()),
        Tensor::empty({shape.flat_topk, shape.intermediate_size}, DataType::I8, hidden_states->device()),
        Tensor::empty({shape.flat_topk, 1}, DataType::F32, hidden_states->device()),
        Tensor::empty({shape.num_tokens, shape.top_k, shape.hidden_size}, hidden_states->dtype(), hidden_states->device()));
}

void deepseek_v4_fused_experts_impl_int8_marlin_impl_(Tensor output,
                                                      const Tensor &hidden_states,
                                                      const Tensor &w1,
                                                      const Tensor &w2,
                                                      const Tensor &topk_weights,
                                                      const Tensor &topk_ids,
                                                      const Tensor &w1_scale,
                                                      const Tensor &w2_scale,
                                                      int64_t global_num_experts,
                                                      double routed_scaling_factor,
                                                      bool inplace,
                                                      const std::optional<Tensor> &shared_output,
                                                      const FusedExpertsWorkspace *workspace) {
    guard_device(hidden_states, "deepseek_v4_fused_experts_impl_int8_marlin_");
    guard_device(output, "deepseek_v4_fused_experts_impl_int8_marlin_");
    if (shared_output.has_value()) {
        guard_device(*shared_output, "deepseek_v4_fused_experts_impl_int8_marlin_");
    }
    check_input_shapes(output, hidden_states, w1, topk_weights, topk_ids, w1_scale, w2_scale, shared_output);
    const auto shape = infer_fused_experts_shape(hidden_states, w1, topk_ids, global_num_experts, workspace != nullptr);
    const int block_size = shape.config.gemm1.block_size_m;

    // Prepare token/expert alignment.
    auto sorted_token_ids = workspace ? workspace->sorted_token_ids : Tensor::empty({shape.max_num_tokens_padded}, DataType::I32, hidden_states->device());
    auto expert_ids = workspace ? workspace->expert_ids : Tensor::empty({(shape.max_num_tokens_padded + static_cast<size_t>(block_size - 1)) / static_cast<size_t>(block_size)}, DataType::I32, hidden_states->device());
    auto num_tokens_post_pad = workspace ? workspace->num_tokens_post_pad : Tensor::empty({1}, DataType::I32, hidden_states->device());
    deepseek_v4_lightop_moe_align_block_size_(
        topk_ids,
        static_cast<int>(shape.num_experts),
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        true);

    // Quantize hidden states.
    auto q_hidden = workspace ? workspace->q_hidden : Tensor::empty(hidden_states->shape(), DataType::I8, hidden_states->device());
    auto hidden_scale = workspace ? workspace->hidden_scale : Tensor::empty({shape.num_tokens, 1}, DataType::F32, hidden_states->device());
    lmslim_per_token_quant_int8_bf16_(q_hidden, hidden_scale, hidden_states);

    // Expert gate/up projection.
    auto gate_up = workspace ? workspace->gate_up : Tensor::empty({shape.num_tokens, shape.top_k, shape.gate_up_size}, hidden_states->dtype(), hidden_states->device());
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
        static_cast<int>(shape.top_k),
        shape.config.gemm1.mode,
        shape.config.gemm1.delta);

    // Activation and dynamic quantization.
    auto q_activated = workspace ? workspace->q_activated : Tensor::empty({shape.flat_topk, shape.intermediate_size}, DataType::I8, hidden_states->device());
    auto activated_scale = workspace ? workspace->activated_scale : Tensor::empty({shape.flat_topk, 1}, DataType::F32, hidden_states->device());
    deepseek_v4_lightop_fuse_silu_mul_quant_(
        q_activated,
        activated_scale,
        gate_up->view({shape.flat_topk, shape.gate_up_size}),
        std::nullopt,
        1,
        -1,
        std::nullopt);

    // Expert down projection.
    auto down = workspace ? workspace->down : Tensor::empty({shape.num_tokens, shape.top_k, shape.hidden_size}, hidden_states->dtype(), hidden_states->device());
    deepseek_v4_lightop_moe_gemm_marlin_w8a8_(
        q_activated,
        w2,
        down,
        activated_scale,
        w2_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        1,
        shape.config.gemm2.mode,
        shape.config.gemm2.delta);

    // Reduce top-k expert outputs and optionally add shared output.
    Tensor target_output = inplace ? hidden_states : output;
    if (shared_output.has_value()) {
        if (hidden_states->dtype() != DataType::BF16) {
            throw std::runtime_error("deepseek_v4_fused_experts_impl_int8_marlin_ shared_output path currently expects BF16 hidden states.");
        }
        deepseek_v4_fused_experts_impl_int8_marlin::launch_moe_sum_scale_add_bf16(
            target_output->data(),
            down->data(),
            (*shared_output)->data(),
            static_cast<int64_t>(shape.num_tokens),
            static_cast<int64_t>(shape.top_k),
            static_cast<int64_t>(shape.hidden_size),
            static_cast<float>(routed_scaling_factor),
            context::getStream());
    } else {
        deepseek_v4_lightop_moe_sum_(
            target_output,
            down,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            static_cast<float>(routed_scaling_factor),
            -1);
    }
    if (!same_storage(target_output, output)) {
        output->copy_from(target_output);
    }
}

} // namespace

DeepseekV4FusedExpertsImplInt8Marlin::DeepseekV4FusedExpertsImplInt8Marlin(Tensor output,
                                                                           const Tensor &hidden_states,
                                                                           const Tensor &w1,
                                                                           const Tensor &w2,
                                                                           const Tensor &topk_weights,
                                                                           const Tensor &topk_ids,
                                                                           const Tensor &w1_scale,
                                                                           const Tensor &w2_scale,
                                                                           int64_t global_num_experts,
                                                                           double routed_scaling_factor,
                                                                           bool inplace,
                                                                           std::optional<Tensor> shared_output) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale);
    if (shared_output.has_value()) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, *shared_output);
    }
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(),
                                 output,
                                 hidden_states,
                                 w1,
                                 w2,
                                 topk_weights,
                                 topk_ids,
                                 w1_scale,
                                 w2_scale,
                                 global_num_experts,
                                 routed_scaling_factor,
                                 inplace,
                                 shared_output);
}

void DeepseekV4FusedExpertsImplInt8Marlin::execute(Tensor output,
                                                   const Tensor &hidden_states,
                                                   const Tensor &w1,
                                                   const Tensor &w2,
                                                   const Tensor &topk_weights,
                                                   const Tensor &topk_ids,
                                                   const Tensor &w1_scale,
                                                   const Tensor &w2_scale,
                                                   int64_t global_num_experts,
                                                   double routed_scaling_factor,
                                                   bool inplace,
                                                   std::optional<Tensor> shared_output) {
    if (fused_experts_graph_debug_enabled()) {
        std::fprintf(stderr,
                     "[infinicore graph] execute DeepseekV4FusedExpertsImplInt8Marlin recording=%d tokens=%zu\n",
                     context::isGraphRecording() ? 1 : 0,
                     hidden_states->size(0));
    }
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4FusedExpertsImplInt8Marlin,
                                      output,
                                      hidden_states,
                                      w1,
                                      w2,
                                      topk_weights,
                                      topk_ids,
                                      w1_scale,
                                      w2_scale,
                                      global_num_experts,
                                      routed_scaling_factor,
                                      inplace,
                                      shared_output);
}

namespace {

struct FusedExpertsGraphMeta {
    graph::GraphTensor output;
    graph::GraphTensor hidden_states;
    graph::GraphTensor w1;
    graph::GraphTensor w2;
    graph::GraphTensor topk_weights;
    graph::GraphTensor topk_ids;
    graph::GraphTensor w1_scale;
    graph::GraphTensor w2_scale;
    FusedExpertsWorkspace workspace;
    std::optional<graph::GraphTensor> shared_output;
    int64_t global_num_experts;
    double routed_scaling_factor;
    bool inplace;

    FusedExpertsGraphMeta(Tensor output_,
                          const Tensor &hidden_states_,
                          const Tensor &w1_,
                          const Tensor &w2_,
                          const Tensor &topk_weights_,
                          const Tensor &topk_ids_,
                          const Tensor &w1_scale_,
                          const Tensor &w2_scale_,
                          int64_t global_num_experts_,
                          double routed_scaling_factor_,
                          bool inplace_,
                          std::optional<Tensor> shared_output_)
        : output(output_),
          hidden_states(hidden_states_),
          w1(w1_),
          w2(w2_),
          topk_weights(topk_weights_),
          topk_ids(topk_ids_),
          w1_scale(w1_scale_),
          w2_scale(w2_scale_),
          workspace(make_workspace(hidden_states_, w1_, topk_ids_, global_num_experts_)),
          global_num_experts(global_num_experts_),
          routed_scaling_factor(routed_scaling_factor_),
          inplace(inplace_) {
        if (shared_output_.has_value()) {
            shared_output.emplace(*shared_output_);
        }
    }
};

void *plan_fused_experts(Tensor output,
                         const Tensor &hidden_states,
                         const Tensor &w1,
                         const Tensor &w2,
                         const Tensor &topk_weights,
                         const Tensor &topk_ids,
                         const Tensor &w1_scale,
                         const Tensor &w2_scale,
                         int64_t global_num_experts,
                         double routed_scaling_factor,
                         bool inplace,
                         std::optional<Tensor> shared_output) {
    guard_device(hidden_states, "DeepseekV4FusedExpertsImplInt8Marlin");
    guard_device(output, "DeepseekV4FusedExpertsImplInt8Marlin");
    guard_device(w1, "DeepseekV4FusedExpertsImplInt8Marlin");
    guard_device(w2, "DeepseekV4FusedExpertsImplInt8Marlin");
    if (shared_output.has_value()) {
        guard_device(*shared_output, "DeepseekV4FusedExpertsImplInt8Marlin");
    }
    check_input_shapes(output, hidden_states, w1, topk_weights, topk_ids, w1_scale, w2_scale, shared_output);
    return new FusedExpertsGraphMeta(output,
                                     hidden_states,
                                     w1,
                                     w2,
                                     topk_weights,
                                     topk_ids,
                                     w1_scale,
                                     w2_scale,
                                     global_num_experts,
                                     routed_scaling_factor,
                                     inplace,
                                     shared_output);
}

void run_fused_experts(void *meta) {
    auto *m = static_cast<FusedExpertsGraphMeta *>(meta);
    std::optional<Tensor> shared_output;
    if (m->shared_output.has_value()) {
        shared_output.emplace(*m->shared_output);
    }
    if (fused_experts_graph_debug_enabled()) {
        std::fprintf(stderr,
                     "[infinicore graph] run DeepseekV4FusedExpertsImplInt8Marlin tokens=%zu\n",
                     m->hidden_states->size(0));
    }
    deepseek_v4_fused_experts_impl_int8_marlin_impl_(m->output,
                                                     m->hidden_states,
                                                     m->w1,
                                                     m->w2,
                                                     m->topk_weights,
                                                     m->topk_ids,
                                                     m->w1_scale,
                                                     m->w2_scale,
                                                     m->global_num_experts,
                                                     m->routed_scaling_factor,
                                                     m->inplace,
                                                     shared_output,
                                                     &m->workspace);
}

void cleanup_fused_experts(void **meta) {
    if (meta && *meta) {
        delete static_cast<FusedExpertsGraphMeta *>(*meta);
        *meta = nullptr;
    }
}

} // namespace

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4FusedExpertsImplInt8Marlin,
                                       &plan_fused_experts,
                                       &run_fused_experts,
                                       &cleanup_fused_experts);

void deepseek_v4_fused_experts_impl_int8_marlin_(Tensor output,
                                                 const Tensor &hidden_states,
                                                 const Tensor &w1,
                                                 const Tensor &w2,
                                                 const Tensor &topk_weights,
                                                 const Tensor &topk_ids,
                                                 const Tensor &w1_scale,
                                                 const Tensor &w2_scale,
                                                 int64_t global_num_experts,
                                                 double routed_scaling_factor,
                                                 bool inplace,
                                                 const std::optional<Tensor> &shared_output) {

    DeepseekV4FusedExpertsImplInt8Marlin::execute(output,
                                                  hidden_states,
                                                  w1,
                                                  w2,
                                                  topk_weights,
                                                  topk_ids,
                                                  w1_scale,
                                                  w2_scale,
                                                  global_num_experts,
                                                  routed_scaling_factor,
                                                  inplace,
                                                  shared_output);
}

} // namespace infinicore::op
