#include "infinicore/ops/deepseek_v4_lmslim_hipblaslt_channelwise_linear_w8a8.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/ops/deepseek_v4_lightop_per_token_dynamic_quant_int8.hpp"

#if defined(ENABLE_HYGON_API)
#include <ATen/ATen.h>
#include <ATen/core/ScalarType.h>
#include <c10/hip/HIPGuard.h>
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4LmslimHipblasltChannelwiseLinearW8A8);

namespace {

#if defined(ENABLE_HYGON_API)
using HipblasltChannelwiseFn = at::Tensor (*)(
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    c10::ScalarType,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    const std::string &,
    const at::Tensor &,
    const at::Tensor &,
    const std::optional<at::Tensor> &);

template <typename Fn>
Fn checked_symbol(void *handle, const char *name) {
    dlerror();
    void *symbol = dlsym(handle, name);
    const char *error = dlerror();
    if (error != nullptr || symbol == nullptr) {
        throw std::runtime_error(std::string("lmslimquant SO is missing required symbol ") + name +
                                 (error != nullptr ? std::string(": ") + error : ""));
    }
    return reinterpret_cast<Fn>(symbol);
}

bool path_exists(const std::string &path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

void *open_lmslimquant_so() {
    std::vector<std::string> candidates;
    if (const char *env_path = std::getenv("INFINICORE_LMSLIMQUANT_SO")) {
        if (env_path[0] != '\0') {
            candidates.emplace_back(env_path);
        }
    }
    candidates.emplace_back("/usr/local/lib/python3.10/dist-packages/lmslimquant.cpython-310-x86_64-linux-gnu.so");
    candidates.emplace_back("/usr/local/lib/python3.11/dist-packages/lmslimquant.cpython-311-x86_64-linux-gnu.so");
    candidates.emplace_back("lmslimquant.cpython-310-x86_64-linux-gnu.so");
    candidates.emplace_back("lmslimquant.cpython-311-x86_64-linux-gnu.so");

    std::ostringstream errors;
    for (const auto &path : candidates) {
        if (path.find('/') != std::string::npos && !path_exists(path)) {
            errors << "\n  " << path << ": not found";
            continue;
        }
        void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (handle != nullptr) {
            return handle;
        }
        if (const char *error = dlerror()) {
            errors << "\n  " << path << ": " << error;
        }
    }
    throw std::runtime_error("failed to load lmslimquant SO. Set INFINICORE_LMSLIMQUANT_SO to lmslimquant*.so." + errors.str());
}

struct LmslimHipblasltSymbols {
    void *handle{nullptr};
    HipblasltChannelwiseFn channelwise{nullptr};
    HipblasltChannelwiseFn channelwise_fast{nullptr};
};

const LmslimHipblasltSymbols &lmslim_hipblaslt_symbols() {
    static LmslimHipblasltSymbols symbols;
    static std::once_flag once;
    std::call_once(once, [] {
        symbols.handle = open_lmslimquant_so();
        symbols.channelwise = checked_symbol<HipblasltChannelwiseFn>(
            symbols.handle,
            "_ZN17hipblaslt_gemm_v226hipblaslt_gemm_channelwiseERKN2at6TensorES3_S3_S3_N3c1010ScalarTypeEllllRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES3_S3_RKSt8optionalIS1_E");
        symbols.channelwise_fast = checked_symbol<HipblasltChannelwiseFn>(
            symbols.handle,
            "_ZN17hipblaslt_gemm_v231hipblaslt_gemm_channelwise_fastERKN2at6TensorES3_S3_S3_N3c1010ScalarTypeEllllRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES3_S3_RKSt8optionalIS1_E");
    });
    return symbols;
}

bool use_fast_channelwise() {
    static const bool enabled = [] {
        const char *value = std::getenv("INFINICORE_DSV4_LMSLIM_HIPBLASLT_CHANNELWISE_FAST");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

void guard_device(const Tensor &tensor, const char *op_name) {
    if (tensor->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error(std::string(op_name) + " expects HYGON tensors in this build.");
    }
}

at::Tensor make_scalar_i32(int value) {
    return at::scalar_tensor(value, at::TensorOptions().dtype(at::kInt).device(at::kCPU));
}
#endif

void check_same_device(const Tensor &base, const Tensor &other, const char *op_name, const char *arg_name) {
    if (base->device() != other->device()) {
        throw std::runtime_error(std::string(op_name) + " expects " + arg_name + " on the same device as output.");
    }
}

bool profile_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("INFINICORE_DSV4_LMSLIM_HIPBLASLT_CHANNELWISE_LINEAR_PROFILE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

void check_tensors(Tensor output,
                   const Tensor &input,
                   const Tensor &weight,
                   const Tensor &weight_scale,
                   const std::optional<Tensor> &bias,
                   Tensor q_input,
                   Tensor input_scale,
                   const Tensor &smooth_scale,
                   const char *op_name) {
    if (output->ndim() != 2 || input->ndim() != 2 || weight->ndim() != 2 ||
        weight_scale->ndim() != 2 || q_input->ndim() != 2 || input_scale->ndim() != 2 ||
        smooth_scale->ndim() != 1) {
        throw std::runtime_error(std::string(op_name) + " expects 2D tensors and 1D smooth_scale.");
    }
    if (output->dtype() != input->dtype() ||
        (output->dtype() != DataType::BF16 && output->dtype() != DataType::F16) ||
        weight->dtype() != DataType::I8 || weight_scale->dtype() != DataType::F32 ||
        q_input->dtype() != DataType::I8 || input_scale->dtype() != DataType::F32 ||
        smooth_scale->dtype() != DataType::F32) {
        throw std::runtime_error(std::string(op_name) + " expects BF16/F16 output/input, I8 weight/q_input, and F32 scales.");
    }
    if (!output->is_contiguous() || !input->is_contiguous() || !weight->is_contiguous() ||
        !weight_scale->is_contiguous() || !q_input->is_contiguous() || !input_scale->is_contiguous() ||
        !smooth_scale->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous output/input/weight/workspaces/scales.");
    }

    check_same_device(output, input, op_name, "input");
    check_same_device(output, weight, op_name, "weight");
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
    const size_t n = weight->size(0);
    if (weight->size(1) != k || output->shape() != std::vector<size_t>{m, n}) {
        throw std::runtime_error(std::string(op_name) + " shape mismatch for input [M,K], weight [N,K], output [M,N].");
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
              const Tensor &weight,
              const Tensor &weight_scale,
              std::optional<Tensor> bias,
              Tensor q_input,
              Tensor input_scale,
              const Tensor &smooth_scale) {
    constexpr const char *op_name = "deepseek_v4_lmslim_hipblaslt_channelwise_linear_w8a8_";
    check_tensors(output, input, weight, weight_scale, bias, q_input, input_scale, smooth_scale, op_name);

#if defined(ENABLE_HYGON_API)
    guard_device(output, op_name);
    const size_t device_index = output->device().getIndex();
    int current_device = -1;
    (void)hipGetDevice(&current_device);
    if (current_device != static_cast<int>(device_index)) {
        const hipError_t status = hipSetDevice(static_cast<int>(device_index));
        if (status != hipSuccess) {
            throw std::runtime_error(std::string(op_name) + " failed to set HIP device: " + hipGetErrorString(status));
        }
    }

    auto run_once = [&] {
        deepseek_v4_lightop_per_token_dynamic_quant_int8_(q_input, input, input_scale, smooth_scale);

        c10::hip::HIPGuard device_guard(static_cast<c10::DeviceIndex>(device_index));
        c10::hip::HIPStreamGuard stream_guard(infinicore::adaptor::get_hip_stream());

        auto output_at = infinicore::adaptor::to_aten_tensor(output);
        auto weight_at = infinicore::adaptor::to_aten_tensor(weight);
        auto q_input_at = infinicore::adaptor::to_aten_tensor(q_input);
        auto weight_scale_at = infinicore::adaptor::to_aten_tensor(weight_scale);
        auto input_scale_at = infinicore::adaptor::to_aten_tensor(input_scale);
        std::optional<at::Tensor> bias_at = std::nullopt;
        if (bias.has_value()) {
            bias_at = infinicore::adaptor::to_aten_tensor(*bias);
        }

        const auto alpha = make_scalar_i32(1);
        const auto beta = make_scalar_i32(0);
        const auto &symbols = lmslim_hipblaslt_symbols();
        auto *fn = use_fast_channelwise() ? symbols.channelwise_fast : symbols.channelwise;
        auto result = fn(weight_at,
                         q_input_at,
                         weight_scale_at,
                         input_scale_at,
                         infinicore::adaptor::to_at_dtype(output->dtype()),
                         static_cast<int64_t>(weight->size(0)),
                         static_cast<int64_t>(input->size(0)),
                         static_cast<int64_t>(input->size(1)),
                         1,
                         std::string("TN"),
                         alpha,
                         beta,
                         bias_at);
        if (result.dim() == 3 && result.size(0) == 1 &&
            result.size(1) == static_cast<int64_t>(input->size(0)) &&
            result.size(2) == static_cast<int64_t>(weight->size(0))) {
            output_at.copy_(result.squeeze(0));
            return;
        }
        if (result.dim() == 2 &&
            result.size(0) == static_cast<int64_t>(input->size(0)) &&
            result.size(1) == static_cast<int64_t>(weight->size(0))) {
            output_at.copy_(result);
            return;
        }
        throw std::runtime_error(std::string(op_name) + " unexpected lmslim hipBLASLt result shape.");
    };

    if (profile_enabled()) {
        context::syncStream();
        const auto total_start = std::chrono::steady_clock::now();
        run_once();
        context::syncStream();
        const auto total_end = std::chrono::steady_clock::now();
        const auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();
        std::fprintf(stderr,
                     "[INFINICORE_DSV4_LMSLIM_HIPBLASLT_CHANNELWISE_LINEAR_PROFILE] M=%zu N=%zu K=%zu total_ms=%.6f fast=%d\n",
                     input->size(0),
                     output->size(1),
                     input->size(1),
                     static_cast<double>(total_us) / 1000.0,
                     use_fast_channelwise() ? 1 : 0);
    } else {
        run_once();
    }
#else
    (void)output;
    (void)input;
    (void)weight;
    (void)weight_scale;
    (void)bias;
    (void)q_input;
    (void)input_scale;
    (void)smooth_scale;
    throw std::runtime_error("deepseek_v4_lmslim_hipblaslt_channelwise_linear_w8a8_ requires a HYGON build.");
#endif
}

} // namespace

DeepseekV4LmslimHipblasltChannelwiseLinearW8A8::DeepseekV4LmslimHipblasltChannelwiseLinearW8A8(Tensor output,
                                                                                               const Tensor &input,
                                                                                               const Tensor &weight,
                                                                                               const Tensor &weight_scale,
                                                                                               std::optional<Tensor> bias,
                                                                                               Tensor q_input,
                                                                                               Tensor input_scale,
                                                                                               const Tensor &smooth_scale) {
    device_graph_capture_supported_ = false;
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(),
                                 output,
                                 input,
                                 weight,
                                 weight_scale,
                                 bias,
                                 q_input,
                                 input_scale,
                                 smooth_scale);
}

void DeepseekV4LmslimHipblasltChannelwiseLinearW8A8::execute(Tensor output,
                                                             const Tensor &input,
                                                             const Tensor &weight,
                                                             const Tensor &weight_scale,
                                                             std::optional<Tensor> bias,
                                                             Tensor q_input,
                                                             Tensor input_scale,
                                                             const Tensor &smooth_scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4LmslimHipblasltChannelwiseLinearW8A8,
                                      output,
                                      input,
                                      weight,
                                      weight_scale,
                                      bias,
                                      q_input,
                                      input_scale,
                                      smooth_scale);
}

namespace deepseek_v4_lmslim_hipblaslt_channelwise_linear_w8a8_graph_impl {

struct PlannedMeta {
    graph::GraphTensor output;
    graph::GraphTensor input;
    graph::GraphTensor weight;
    graph::GraphTensor weight_scale;
    std::optional<graph::GraphTensor> bias;
    graph::GraphTensor q_input;
    graph::GraphTensor input_scale;
    graph::GraphTensor smooth_scale;
};

void *plan(Tensor output,
           const Tensor &input,
           const Tensor &weight,
           const Tensor &weight_scale,
           std::optional<Tensor> bias,
           Tensor q_input,
           Tensor input_scale,
           const Tensor &smooth_scale) {
    check_tensors(output,
                  input,
                  weight,
                  weight_scale,
                  bias,
                  q_input,
                  input_scale,
                  smooth_scale,
                  "deepseek_v4_lmslim_hipblaslt_channelwise_linear_w8a8_");
    std::optional<graph::GraphTensor> graph_bias = std::nullopt;
    if (bias.has_value()) {
        graph_bias = graph::GraphTensor(bias.value());
    }
    return new PlannedMeta{graph::GraphTensor(output),
                           graph::GraphTensor(input),
                           graph::GraphTensor(weight),
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
             planned->weight,
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

} // namespace deepseek_v4_lmslim_hipblaslt_channelwise_linear_w8a8_graph_impl

namespace deepseek_v4_lmslim_hipblaslt_channelwise_linear_w8a8_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4LmslimHipblasltChannelwiseLinearW8A8,
                                       &deepseek_v4_lmslim_hipblaslt_channelwise_linear_w8a8_graph_impl::plan,
                                       &deepseek_v4_lmslim_hipblaslt_channelwise_linear_w8a8_graph_impl::run,
                                       &deepseek_v4_lmslim_hipblaslt_channelwise_linear_w8a8_graph_impl::cleanup);
} // namespace deepseek_v4_lmslim_hipblaslt_channelwise_linear_w8a8_register

void deepseek_v4_lmslim_hipblaslt_channelwise_linear_w8a8_(Tensor output,
                                                           const Tensor &input,
                                                           const Tensor &weight,
                                                           const Tensor &weight_scale,
                                                           std::optional<Tensor> bias,
                                                           Tensor q_input,
                                                           Tensor input_scale,
                                                           const Tensor &smooth_scale) {
    DeepseekV4LmslimHipblasltChannelwiseLinearW8A8::execute(output,
                                                            input,
                                                            weight,
                                                            weight_scale,
                                                            bias,
                                                            q_input,
                                                            input_scale,
                                                            smooth_scale);
}

} // namespace infinicore::op
