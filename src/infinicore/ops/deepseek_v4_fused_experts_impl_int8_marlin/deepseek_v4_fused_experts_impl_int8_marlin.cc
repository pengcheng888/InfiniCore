#include "infinicore/ops/deepseek_v4_fused_experts_impl_int8_marlin.hpp"

#include "infinicore/device.hpp"
#include "deepseek_v4_fused_experts_impl_int8_marlin_kernel.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/ops/deepseek_v4_lightop_moe_marlin.hpp"

#include <optional>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace infinicore::op {

namespace {

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

struct MarlinConfig {
    int block_size{16};
    int gemm1_mode{54};
    int gemm2_mode{54};
    int delta{1};
    bool supported{false};
};

MarlinConfig select_deepseek_v4_marlin_config(size_t num_tokens,
                                              size_t hidden_size,
                                              size_t intermediate_size,
                                              size_t top_k) {
    MarlinConfig config;
    if (hidden_size == 7168 && intermediate_size == 256 && top_k == 8) {
        config.supported = true;
        if (num_tokens <= 1) {
            config.gemm1_mode = 21;
            config.gemm2_mode = 25;
        } else if (num_tokens <= 7) {
            config.gemm1_mode = 78;
            config.gemm2_mode = 73;
        } else if (num_tokens <= 16) {
            config.gemm1_mode = 29;
            config.gemm2_mode = 12;
        } else if (num_tokens <= 75) {
            config.gemm1_mode = 55;
            config.gemm2_mode = 54;
        }
        return config;
    }
    if (hidden_size == 4096 && intermediate_size == 256 && top_k == 6) {
        config.supported = true;
        if (num_tokens <= 1) {
            config.gemm1_mode = 58;
            config.gemm2_mode = 16;
        } else if (num_tokens <= 7) {
            config.gemm1_mode = 29;
            config.gemm2_mode = 9;
        } else if (num_tokens <= 16) {
            config.gemm1_mode = 29;
            config.gemm2_mode = 4;
        } else if (num_tokens <= 75) {
            config.gemm1_mode = 29;
            config.gemm2_mode = 12;
        } else {
            config.gemm1_mode = 37;
            config.gemm2_mode = 54;
        }
        return config;
    }
    return config;
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

void *current_accelerator_stream() {
#if defined(ENABLE_HYGON_API)
    return reinterpret_cast<void *>(infinicore::adaptor::get_hip_stream().stream());
#else
    return reinterpret_cast<void *>(infinicore::adaptor::get_cuda_stream().stream());
#endif
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
        current_accelerator_stream());
}

} // namespace

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
    guard_device(hidden_states, "deepseek_v4_fused_experts_impl_int8_marlin_");
    guard_device(output, "deepseek_v4_fused_experts_impl_int8_marlin_");
    if (shared_output.has_value()) {
        guard_device(*shared_output, "deepseek_v4_fused_experts_impl_int8_marlin_");
    }
    check_input_shapes(output, hidden_states, w1, topk_weights, topk_ids, w1_scale, w2_scale, shared_output);

    const size_t num_tokens = hidden_states->size(0);
    const size_t hidden_size = hidden_states->size(1);
    const size_t top_k = topk_ids->size(1);
    const size_t num_experts = static_cast<size_t>(global_num_experts > 0 ? global_num_experts : static_cast<int64_t>(w1->size(0)));
    const size_t intermediate_size = w1->size(2) / 128;
    const size_t gate_up_size = intermediate_size * 2;
    const size_t flat_topk = num_tokens * top_k;
    const auto config = select_deepseek_v4_marlin_config(num_tokens, hidden_size, intermediate_size, top_k);
    if (!config.supported) {
        throw std::runtime_error("deepseek_v4_fused_experts_impl_int8_marlin_ only supports DeepSeek-V4 Marlin shapes hidden=4096/topk=6/local_intermediate=256 or hidden=7168/topk=8/local_intermediate=256.");
    }

    const int block_size = config.block_size;
    size_t max_num_tokens_padded = flat_topk + num_experts * static_cast<size_t>(block_size - 1);
    max_num_tokens_padded = ((max_num_tokens_padded + static_cast<size_t>(block_size - 1)) / static_cast<size_t>(block_size)) * static_cast<size_t>(block_size);
    if (flat_topk < num_experts) {
        max_num_tokens_padded = std::min(flat_topk * static_cast<size_t>(block_size), max_num_tokens_padded);
    }

    auto sorted_token_ids = Tensor::empty({max_num_tokens_padded}, DataType::I32, hidden_states->device());
    auto expert_ids = Tensor::empty({(max_num_tokens_padded + static_cast<size_t>(block_size - 1)) / static_cast<size_t>(block_size)}, DataType::I32, hidden_states->device());
    auto num_tokens_post_pad = Tensor::empty({1}, DataType::I32, hidden_states->device());
    deepseek_v4_lightop_moe_align_block_size_(
        topk_ids,
        static_cast<int>(num_experts),
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        true);

    auto q_hidden = Tensor::empty(hidden_states->shape(), DataType::I8, hidden_states->device());
    auto hidden_scale = Tensor::empty({num_tokens, 1}, DataType::F32, hidden_states->device());
    lmslim_per_token_quant_int8_bf16_(q_hidden, hidden_scale, hidden_states);

    auto gate_up = Tensor::empty({num_tokens, top_k, gate_up_size}, hidden_states->dtype(), hidden_states->device());
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
        static_cast<int>(top_k),
        config.gemm1_mode,
        config.delta);

    auto q_activated = Tensor::empty({flat_topk, intermediate_size}, DataType::I8, hidden_states->device());
    auto activated_scale = Tensor::empty({flat_topk, 1}, DataType::F32, hidden_states->device());
    deepseek_v4_lightop_fuse_silu_mul_quant_(
        q_activated,
        activated_scale,
        gate_up->view({flat_topk, gate_up_size}),
        std::nullopt,
        1,
        -1,
        std::nullopt);

    auto down = Tensor::empty({num_tokens, top_k, hidden_size}, hidden_states->dtype(), hidden_states->device());
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
        config.gemm2_mode,
        config.delta);

    Tensor target_output = inplace ? hidden_states : output;
    if (shared_output.has_value()) {
        if (hidden_states->dtype() != DataType::BF16) {
            throw std::runtime_error("deepseek_v4_fused_experts_impl_int8_marlin_ shared_output path currently expects BF16 hidden states.");
        }
        deepseek_v4_fused_experts_impl_int8_marlin::launch_moe_sum_scale_add_bf16(
            target_output->data(),
            down->data(),
            (*shared_output)->data(),
            static_cast<int64_t>(num_tokens),
            static_cast<int64_t>(top_k),
            static_cast<int64_t>(hidden_size),
            static_cast<float>(routed_scaling_factor),
            current_accelerator_stream());
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

} // namespace infinicore::op
