#include "infinicore/ops/deepseek_v4_lmslim_linear_w8a8.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops/deepseek_v4_lightop_per_token_dynamic_quant_int8.hpp"
#include "infinicore/ops/scaled_mm_i8.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4LmslimLinearW8A8);

namespace {

void check_same_device(const Tensor &base, const Tensor &other, const char *op_name, const char *arg_name) {
    if (base->device() != other->device()) {
        throw std::runtime_error(std::string(op_name) + " expects " + arg_name + " on the same device as output.");
    }
}

bool profile_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("INFINICORE_DSV4_LMSLIM_LINEAR_PROFILE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

void check_tensors(Tensor output,
                   const Tensor &input,
                   const Tensor &weight_t,
                   const Tensor &weight_scale,
                   const std::optional<Tensor> &bias,
                   Tensor q_input,
                   Tensor input_scale,
                   const Tensor &smooth_scale,
                   const char *op_name) {
    if (output->ndim() != 2 || input->ndim() != 2 || weight_t->ndim() != 2 ||
        weight_scale->ndim() != 2 || q_input->ndim() != 2 || input_scale->ndim() != 2 ||
        smooth_scale->ndim() != 1) {
        throw std::runtime_error(std::string(op_name) + " expects 2D tensors and 1D smooth_scale.");
    }
    if (output->dtype() != input->dtype() ||
        (output->dtype() != DataType::BF16 && output->dtype() != DataType::F16) ||
        weight_t->dtype() != DataType::I8 || weight_scale->dtype() != DataType::F32 ||
        q_input->dtype() != DataType::I8 || input_scale->dtype() != DataType::F32 ||
        smooth_scale->dtype() != DataType::F32) {
        throw std::runtime_error(std::string(op_name) + " expects BF16/F16 output/input, I8 weight_t/q_input, and F32 scales.");
    }
    if (!output->is_contiguous() || !input->is_contiguous() ||
        !weight_scale->is_contiguous() || !q_input->is_contiguous() || !input_scale->is_contiguous() ||
        !smooth_scale->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous output/input/workspaces/scales.");
    }
    if (weight_t->stride(0) != 1) {
        throw std::runtime_error(std::string(op_name) + " expects weight_t [K,N] with contiguous K dimension.");
    }

    check_same_device(output, input, op_name, "input");
    check_same_device(output, weight_t, op_name, "weight_t");
    check_same_device(output, weight_scale, op_name, "weight_scale");
    check_same_device(output, q_input, op_name, "q_input");
    check_same_device(output, input_scale, op_name, "input_scale");
    check_same_device(output, smooth_scale, op_name, "smooth_scale");
    if (bias.has_value()) {
        check_same_device(output, bias.value(), op_name, "bias");
        if (bias.value()->ndim() != 1 || bias.value()->dtype() != output->dtype() ||
            !bias.value()->is_contiguous()) {
            throw std::runtime_error(std::string(op_name) + " expects contiguous 1D bias with output dtype.");
        }
    }

    const size_t m = input->size(0);
    const size_t k = input->size(1);
    const size_t n = weight_t->size(1);
    if (weight_t->size(0) != k || output->shape() != std::vector<size_t>{m, n}) {
        throw std::runtime_error(std::string(op_name) + " shape mismatch for input [M,K], weight_t [K,N], output [M,N].");
    }
    if (q_input->shape() != input->shape() || input_scale->shape() != std::vector<size_t>{m, 1}) {
        throw std::runtime_error(std::string(op_name) + " workspace shape mismatch.");
    }
    if (weight_scale->size(0) != n || weight_scale->size(1) != 1) {
        throw std::runtime_error(std::string(op_name) + " expects weight_scale [N,1].");
    }
    if (smooth_scale->size(0) != k) {
        throw std::runtime_error(std::string(op_name) + " expects smooth_scale [K].");
    }
    if (bias.has_value() && bias.value()->size(0) != n) {
        throw std::runtime_error(std::string(op_name) + " bias shape mismatch.");
    }
}

void run_impl(Tensor output,
              const Tensor &input,
              const Tensor &weight_t,
              const Tensor &weight_scale,
              std::optional<Tensor> bias,
              Tensor q_input,
              Tensor input_scale,
              const Tensor &smooth_scale) {
    constexpr const char *op_name = "deepseek_v4_lmslim_linear_w8a8_";
    check_tensors(output, input, weight_t, weight_scale, bias, q_input, input_scale, smooth_scale, op_name);

    if (profile_enabled()) {
        context::syncStream();
        const auto total_start = std::chrono::steady_clock::now();
        const auto quant_start = total_start;
        deepseek_v4_lightop_per_token_dynamic_quant_int8_(q_input, input, input_scale, smooth_scale);
        context::syncStream();
        const auto quant_end = std::chrono::steady_clock::now();
        scaled_mm_i8_(output, q_input, input_scale, weight_t, weight_scale, bias);
        context::syncStream();
        const auto total_end = std::chrono::steady_clock::now();
        const auto quant_us = std::chrono::duration_cast<std::chrono::microseconds>(quant_end - quant_start).count();
        const auto gemm_us = std::chrono::duration_cast<std::chrono::microseconds>(total_end - quant_end).count();
        const auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();
        std::fprintf(stderr,
                     "[INFINICORE_DSV4_LMSLIM_LINEAR_PROFILE] M=%zu N=%zu K=%zu quant_ms=%.6f scaled_mm_ms=%.6f total_ms=%.6f\n",
                     input->size(0),
                     output->size(1),
                     input->size(1),
                     static_cast<double>(quant_us) / 1000.0,
                     static_cast<double>(gemm_us) / 1000.0,
                     static_cast<double>(total_us) / 1000.0);
        return;
    }

    deepseek_v4_lightop_per_token_dynamic_quant_int8_(q_input, input, input_scale, smooth_scale);
    scaled_mm_i8_(output, q_input, input_scale, weight_t, weight_scale, bias);
}

} // namespace

DeepseekV4LmslimLinearW8A8::DeepseekV4LmslimLinearW8A8(Tensor output,
                                                       const Tensor &input,
                                                       const Tensor &weight_t,
                                                       const Tensor &weight_scale,
                                                       std::optional<Tensor> bias,
                                                       Tensor q_input,
                                                       Tensor input_scale,
                                                       const Tensor &smooth_scale) {
    device_graph_capture_supported_ = false;
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(),
                                 output,
                                 input,
                                 weight_t,
                                 weight_scale,
                                 bias,
                                 q_input,
                                 input_scale,
                                 smooth_scale);
}

void DeepseekV4LmslimLinearW8A8::execute(Tensor output,
                                         const Tensor &input,
                                         const Tensor &weight_t,
                                         const Tensor &weight_scale,
                                         std::optional<Tensor> bias,
                                         Tensor q_input,
                                         Tensor input_scale,
                                         const Tensor &smooth_scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4LmslimLinearW8A8,
                                      output,
                                      input,
                                      weight_t,
                                      weight_scale,
                                      bias,
                                      q_input,
                                      input_scale,
                                      smooth_scale);
}

namespace deepseek_v4_lmslim_linear_w8a8_graph_impl {

struct PlannedMeta {
    graph::GraphTensor output;
    graph::GraphTensor input;
    graph::GraphTensor weight_t;
    graph::GraphTensor weight_scale;
    std::optional<graph::GraphTensor> bias;
    graph::GraphTensor q_input;
    graph::GraphTensor input_scale;
    graph::GraphTensor smooth_scale;
};

void *plan(Tensor output,
           const Tensor &input,
           const Tensor &weight_t,
           const Tensor &weight_scale,
           std::optional<Tensor> bias,
           Tensor q_input,
           Tensor input_scale,
           const Tensor &smooth_scale) {
    check_tensors(output, input, weight_t, weight_scale, bias, q_input, input_scale, smooth_scale, "deepseek_v4_lmslim_linear_w8a8_");
    std::optional<graph::GraphTensor> graph_bias = std::nullopt;
    if (bias.has_value()) {
        graph_bias = graph::GraphTensor(bias.value());
    }
    return new PlannedMeta{graph::GraphTensor(output),
                           graph::GraphTensor(input),
                           graph::GraphTensor(weight_t),
                           graph::GraphTensor(weight_scale),
                           graph_bias,
                           graph::GraphTensor(q_input),
                           graph::GraphTensor(input_scale),
                           graph::GraphTensor(smooth_scale)};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    std::optional<Tensor> bias = std::nullopt;
    if (planned->bias.has_value()) {
        bias = planned->bias.value();
    }
    run_impl(planned->output,
             planned->input,
             planned->weight_t,
             planned->weight_scale,
             bias,
             planned->q_input,
             planned->input_scale,
             planned->smooth_scale);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_lmslim_linear_w8a8_graph_impl

namespace deepseek_v4_lmslim_linear_w8a8_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4LmslimLinearW8A8,
                                       &deepseek_v4_lmslim_linear_w8a8_graph_impl::plan,
                                       &deepseek_v4_lmslim_linear_w8a8_graph_impl::run,
                                       &deepseek_v4_lmslim_linear_w8a8_graph_impl::cleanup);
} // namespace deepseek_v4_lmslim_linear_w8a8_register

void deepseek_v4_lmslim_linear_w8a8_(Tensor output,
                                     const Tensor &input,
                                     const Tensor &weight_t,
                                     const Tensor &weight_scale,
                                     std::optional<Tensor> bias,
                                     Tensor q_input,
                                     Tensor input_scale,
                                     const Tensor &smooth_scale) {
    DeepseekV4LmslimLinearW8A8::execute(output, input, weight_t, weight_scale, bias, q_input, input_scale, smooth_scale);
}

} // namespace infinicore::op
