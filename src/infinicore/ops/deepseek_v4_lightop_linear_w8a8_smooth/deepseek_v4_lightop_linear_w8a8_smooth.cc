#include "infinicore/ops/deepseek_v4_lightop_linear_w8a8_smooth.hpp"

#include "deepseek_v4_lightop_linear_w8a8_smooth_kernel.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ATen.h>
#include <ATen/core/ScalarType.h>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>
#elif defined(ENABLE_NVIDIA_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <cmath>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4LightopLinearW8A8Smooth);

namespace {

#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
void guard_device(const Tensor &tensor, const char *op_name) {
#if defined(ENABLE_HYGON_API)
    if (tensor->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error(std::string(op_name) + " expects HYGON tensors in this build.");
    }
#else
    if (tensor->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error(std::string(op_name) + " expects NVIDIA tensors in this build.");
    }
#endif
}

template <typename Fn>
Fn checked_symbol(void *handle, const char *name) {
    dlerror();
    void *symbol = dlsym(handle, name);
    const char *error = dlerror();
    if (error != nullptr || symbol == nullptr) {
        throw std::runtime_error(std::string("lightop SO is missing required symbol ") + name +
                                 (error != nullptr ? std::string(": ") + error : ""));
    }
    return reinterpret_cast<Fn>(symbol);
}

bool path_exists(const std::string &path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

#if defined(ENABLE_HYGON_API)
std::string infer_lightop_gpu_target(int device_index) {
    if (const char *env_target = std::getenv("LIGHTOP_GPU_TARGET")) {
        if (env_target[0] != '\0') {
            return env_target;
        }
    }
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, device_index) == hipSuccess && prop.gcnArchName[0] != '\0') {
        return prop.gcnArchName;
    }
    return "gfx938";
}

std::string infer_lightop_gpu_cus(int device_index) {
    if (const char *env_cus = std::getenv("LIGHTOP_GPU_CUS")) {
        if (env_cus[0] != '\0') {
            return env_cus;
        }
    }
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, device_index) == hipSuccess && prop.multiProcessorCount > 0) {
        return std::to_string(prop.multiProcessorCount);
    }
    return "80";
}

void configure_lightop_env(const std::string &op_so_path, int device_index) {
    const auto slash = op_so_path.find_last_of('/');
    const std::string root = slash == std::string::npos ? "." : op_so_path.substr(0, slash);
    const std::string target = infer_lightop_gpu_target(device_index);
    const std::string asm_dir = root + "/hsa/" + target + "/";
    if (path_exists(asm_dir)) {
        setenv("LIGHTOP_ASM_DIR", asm_dir.c_str(), 0);
    }
    setenv("LIGHTOP_GPU_TARGET", target.c_str(), 0);
    const std::string cus = infer_lightop_gpu_cus(device_index);
    setenv("LIGHTOP_GPU_CUS", cus.c_str(), 0);
}
#endif

void *open_lightop_so() {
    std::vector<std::string> candidates;
    if (const char *env_path = std::getenv("INFINICORE_LIGHTOP_OP_SO")) {
        if (env_path[0] != '\0') {
            candidates.emplace_back(env_path);
        }
    }
    candidates.emplace_back("/usr/local/lib/python3.10/dist-packages/lightop/op.cpython-310-x86_64-linux-gnu.so");
    candidates.emplace_back("/usr/local/lib/python3.11/dist-packages/lightop/op.cpython-311-x86_64-linux-gnu.so");
    candidates.emplace_back("op.cpython-310-x86_64-linux-gnu.so");
    candidates.emplace_back("op.cpython-311-x86_64-linux-gnu.so");

    std::ostringstream errors;
    for (const auto &path : candidates) {
#if defined(ENABLE_HYGON_API)
        int device_index = -1;
        (void)hipGetDevice(&device_index);
        configure_lightop_env(path, device_index < 0 ? 0 : device_index);
#endif
        void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (handle != nullptr) {
            return handle;
        }
        if (const char *error = dlerror()) {
            errors << "\n  " << path << ": " << error;
        }
    }
    throw std::runtime_error("failed to load lightop op SO. Set INFINICORE_LIGHTOP_OP_SO to lightop/op*.so." + errors.str());
}

struct LightopLinearSymbols {
    using GemmW8A8SmoothAsmFn = at::Tensor (*)(at::Tensor &,
                                               at::Tensor &,
                                               at::Tensor &,
                                               at::Tensor &,
                                               std::optional<at::Tensor>,
                                               c10::ScalarType,
                                               int,
                                               int,
                                               int,
                                               bool,
                                               int);
    using PerTokenDynamicQuantInt8Fn = void (*)(at::Tensor &, const at::Tensor &, at::Tensor &, const at::Tensor &);

    void *handle{nullptr};
    GemmW8A8SmoothAsmFn gemm_w8a8_smooth_asm{nullptr};
    PerTokenDynamicQuantInt8Fn per_token_dynamic_quant_int8{nullptr};
};

const LightopLinearSymbols &lightop_linear_symbols() {
    static LightopLinearSymbols symbols;
    static std::once_flag once;
    std::call_once(once, [] {
        symbols.handle = open_lightop_so();
        symbols.gemm_w8a8_smooth_asm = checked_symbol<LightopLinearSymbols::GemmW8A8SmoothAsmFn>(
            symbols.handle,
            "_ZN2at6native20gemm_w8a8_smooth_asmERNS_6TensorES2_S2_S2_St8optionalIS1_EN3c1010ScalarTypeEiiibi");
        symbols.per_token_dynamic_quant_int8 =
            checked_symbol<LightopLinearSymbols::PerTokenDynamicQuantInt8Fn>(
                symbols.handle,
                "_ZN2at6native28per_token_dynamic_quant_int8ERNS_6TensorERKS1_S2_S4_");
    });
    return symbols;
}

std::mutex &lightop_call_mutex() {
    static std::mutex mutex;
    return mutex;
}

bool debug_enabled() {
    static const bool enabled = []() {
        const char *value = std::getenv("INFINICORE_LIGHTOP_LINEAR_DEBUG");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

bool env_enabled(const char *name) {
    const char *value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool disable_lightop_asm() {
    static const bool enabled = env_enabled("INFINICORE_LIGHTOP_LINEAR_DISABLE_ASM");
    return enabled;
}

bool debug_sync_enabled() {
    static const bool enabled = env_enabled("INFINICORE_LIGHTOP_LINEAR_DEBUG_SYNC");
    return enabled;
}

#if defined(ENABLE_HYGON_API)
void debug_check_hip(const char *stage, const Tensor &output) {
    if (!debug_enabled() && !debug_sync_enabled()) {
        return;
    }

    hipError_t sync_status = hipSuccess;
    if (debug_sync_enabled()) {
        sync_status = hipStreamSynchronize(reinterpret_cast<hipStream_t>(context::getStream()));
    }
    hipError_t last_status = hipGetLastError();
    if (debug_enabled()) {
        std::fprintf(stderr,
                     "[deepseek_v4_lightop_linear_w8a8_smooth_] stage=%s output_device=%zu sync=%d(%s) last=%d(%s)\n",
                     stage,
                     output->device().getIndex(),
                     static_cast<int>(sync_status),
                     hipGetErrorString(sync_status),
                     static_cast<int>(last_status),
                     hipGetErrorString(last_status));
    }
    if (sync_status != hipSuccess) {
        throw std::runtime_error(std::string("deepseek_v4_lightop_linear_w8a8_smooth_ HIP sync failed after ") +
                                 stage + ": " + hipGetErrorString(sync_status));
    }
    if (last_status != hipSuccess) {
        throw std::runtime_error(std::string("deepseek_v4_lightop_linear_w8a8_smooth_ HIP error after ") +
                                 stage + ": " + hipGetErrorString(last_status));
    }
}
#else
void debug_check_hip(const char *, const Tensor &) {}
#endif

bool can_use_lightop_asm(int64_t m, int64_t k, const std::optional<at::Tensor> &bias) {
    // Match lightop.gemmopt.gemm_w8a8_smooth: the asm path is intended for M > 16,
    // no bias, and reasonably large K. Small decode shapes should not force asm.
    return m > 16 && k >= 128 && !bias.has_value();
}

int64_t round_up_to_next_power_of_2(int64_t value) {
    if (value <= 1) {
        return 1;
    }
    --value;
    for (int shift = 1; shift < 63; shift <<= 1) {
        value |= value >> shift;
    }
    return value + 1;
}

std::pair<bool, int> select_smooth_asm_config(int64_t m, int64_t n, int64_t k) {
    const int64_t adapt_m = m < 256 ? ((m + 15) / 16) * 16 : round_up_to_next_power_of_2(m);
    const std::string key = "M" + std::to_string(adapt_m) +
                            "N" + std::to_string(n) +
                            "K" + std::to_string(k);

    static const std::unordered_map<std::string, int> tuned = {
        // These entries mirror lightop.config_smooth_gemm for common DeepSeek-V4
        // projection shapes. Missing shapes deliberately fall back to the
        // untuned lightop default, matching lightop.gemmopt.w8a8_smooth_asm_config.
        {"M32N4096K4096", 2},   {"M48N4096K4096", 3},   {"M64N4096K4096", 3},
        {"M80N4096K4096", 10},  {"M96N4096K4096", 10},  {"M112N4096K4096", 3},
        {"M128N4096K4096", 6},  {"M144N4096K4096", 6},  {"M160N4096K4096", 6},
        {"M176N4096K4096", 6},  {"M192N4096K4096", 5},  {"M208N4096K4096", 6},
        {"M224N4096K4096", 6},  {"M240N4096K4096", 6},  {"M256N4096K4096", 6},
        {"M512N4096K4096", 7},  {"M1024N4096K4096", 8}, {"M2048N4096K4096", 8},
        {"M4096N4096K4096", 8}, {"M8192N4096K4096", 8}, {"M16384N4096K4096", 8},

        {"M32N2048K4096", 1},   {"M48N2048K4096", 2},   {"M64N2048K4096", 2},
        {"M80N2048K4096", 3},   {"M96N2048K4096", 3},   {"M112N2048K4096", 3},
        {"M128N2048K4096", 3},  {"M144N2048K4096", 10}, {"M160N2048K4096", 10},
        {"M176N2048K4096", 10}, {"M192N2048K4096", 10}, {"M208N2048K4096", 3},
        {"M224N2048K4096", 3},  {"M240N2048K4096", 3},  {"M256N2048K4096", 6},
        {"M512N2048K4096", 6},  {"M1024N2048K4096", 7}, {"M2048N2048K4096", 8},
        {"M4096N2048K4096", 8}, {"M8192N2048K4096", 8}, {"M16384N2048K4096", 8},

        {"M32N1280K4096", 1},   {"M48N1280K4096", 1},   {"M64N1280K4096", 1},
        {"M80N1280K4096", 2},   {"M96N1280K4096", 2},   {"M112N1280K4096", 2},
        {"M128N1280K4096", 2},  {"M144N1280K4096", 3},  {"M160N1280K4096", 3},
        {"M176N1280K4096", 3},  {"M192N1280K4096", 3},  {"M208N1280K4096", 3},
        {"M224N1280K4096", 3},  {"M240N1280K4096", 3},  {"M256N1280K4096", 3},
        {"M512N1280K4096", 6},  {"M1024N1280K4096", 6}, {"M2048N1280K4096", 7},
        {"M4096N1280K4096", 8}, {"M8192N1280K4096", 8}, {"M16384N1280K4096", 8},

        {"M32N768K4096", 1},    {"M48N768K4096", 1},    {"M64N768K4096", 1},
        {"M80N768K4096", 1},    {"M96N768K4096", 1},    {"M112N768K4096", 2},
        {"M128N768K4096", 2},   {"M144N768K4096", 2},   {"M160N768K4096", 2},
        {"M176N768K4096", 2},   {"M192N768K4096", 2},   {"M208N768K4096", 3},
        {"M224N768K4096", 3},   {"M240N768K4096", 3},   {"M256N768K4096", 3},
        {"M512N768K4096", 10},  {"M1024N768K4096", 6},  {"M2048N768K4096", 7},
        {"M4096N768K4096", 8},  {"M8192N768K4096", 8},  {"M16384N768K4096", 8},
    };

    auto it = tuned.find(key);
    if (it == tuned.end()) {
        return {false, 255};
    }
    return {true, it->second};
}

void compute_reference_from_quantized(at::Tensor &output,
                                      const at::Tensor &q_input,
                                      const at::Tensor &weight,
                                      const at::Tensor &input_scale,
                                      const at::Tensor &weight_scale,
                                      const std::optional<at::Tensor> &bias) {
    auto result = at::matmul(q_input.to(at::kFloat), weight.to(at::kFloat).transpose(0, 1));
    result.mul_(input_scale.to(at::kFloat));
    result.mul_(weight_scale.to(at::kFloat).reshape({1, weight_scale.size(0)}));
    if (bias.has_value()) {
        result.add_(bias->to(at::kFloat).reshape({1, bias->size(0)}));
    }
    output.copy_(result.to(output.scalar_type()));
}

void compute_native_from_quantized(Tensor output,
                                   const Tensor &q_input,
                                   const Tensor &weight,
                                   const Tensor &input_scale,
                                   const Tensor &weight_scale,
                                   const std::optional<Tensor> &bias) {
    deepseek_v4_lightop_linear_w8a8_smooth_impl::launch_w8a8_smooth_gemm_bf16(
        output->data(),
        reinterpret_cast<const int8_t *>(q_input->data()),
        reinterpret_cast<const int8_t *>(weight->data()),
        reinterpret_cast<const float *>(input_scale->data()),
        reinterpret_cast<const float *>(weight_scale->data()),
        bias.has_value() ? (*bias)->data() : nullptr,
        static_cast<int64_t>(q_input->size(0)),
        static_cast<int64_t>(weight->size(0)),
        static_cast<int64_t>(q_input->size(1)),
        context::getStream());
}

void check_same_device(const Tensor &expected,
                       const Tensor &actual,
                       const char *op_name,
                       const char *arg_name) {
    if (actual->device() != expected->device()) {
        throw std::runtime_error(std::string(op_name) + " expects " + arg_name + " on the same device as output.");
    }
}

void check_tensors(const Tensor &input,
                   const Tensor &weight,
                   const Tensor &weight_scale,
                   const std::optional<Tensor> &bias,
                   const Tensor &q_input,
                   const Tensor &input_scale,
                   const Tensor &smooth_scale,
                   const char *op_name) {
    guard_device(input, op_name);
    if (input->ndim() != 2 || weight->ndim() != 2 || weight_scale->ndim() != 2 ||
        q_input->ndim() != 2 || input_scale->ndim() != 2 || smooth_scale->ndim() != 1) {
        throw std::runtime_error(std::string(op_name) + " expects input/weight/workspace 2D tensors and smooth_scale 1D.");
    }
    if (input->dtype() != DataType::BF16 || weight->dtype() != DataType::I8 ||
        weight_scale->dtype() != DataType::F32 || q_input->dtype() != DataType::I8 ||
        input_scale->dtype() != DataType::F32 || smooth_scale->dtype() != DataType::F32) {
        throw std::runtime_error(std::string(op_name) + " unexpected dtype.");
    }
    if (!input->is_contiguous() || !weight->is_contiguous() || !weight_scale->is_contiguous() ||
        !q_input->is_contiguous() || !input_scale->is_contiguous() || !smooth_scale->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous input, weight, scales, and workspace.");
    }
    const size_t m = input->size(0);
    const size_t k = input->size(1);
    const size_t n = weight->size(0);
    if (weight->size(1) != k || weight_scale->size(0) != n || weight_scale->size(1) != 1 ||
        q_input->shape() != input->shape() || input_scale->shape() != std::vector<size_t>{m, 1} ||
        smooth_scale->size(0) != k) {
        throw std::runtime_error(std::string(op_name) + " shape mismatch.");
    }
    if (bias.has_value()) {
        guard_device(*bias, op_name);
        if ((*bias)->dtype() != input->dtype() || (*bias)->ndim() != 1 || (*bias)->size(0) != n) {
            throw std::runtime_error(std::string(op_name) + " bias shape or dtype mismatch.");
        }
    }
}

void check_output(const Tensor &output,
                  const Tensor &input,
                  const Tensor &weight,
                  const char *op_name) {
    guard_device(output, op_name);
    if (output->ndim() != 2 ||
        output->size(0) != input->size(0) ||
        output->size(1) != weight->size(0) ||
        output->dtype() != input->dtype() ||
        output->device() != input->device()) {
        throw std::runtime_error(std::string(op_name) + " output shape, dtype, or device mismatch.");
    }
    if (!output->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous output.");
    }
    check_same_device(output, weight, op_name, "weight");
    check_same_device(output, input, op_name, "input");
}

void check_workspace_devices(const Tensor &output,
                             const Tensor &weight_scale,
                             const std::optional<Tensor> &bias,
                             const Tensor &q_input,
                             const Tensor &input_scale,
                             const Tensor &smooth_scale,
                             const char *op_name) {
    check_same_device(output, weight_scale, op_name, "weight_scale");
    check_same_device(output, q_input, op_name, "q_input");
    check_same_device(output, input_scale, op_name, "input_scale");
    check_same_device(output, smooth_scale, op_name, "smooth_scale");
    if (bias.has_value()) {
        check_same_device(output, *bias, op_name, "bias");
    }
}

void check_all_tensors(Tensor output,
                       const Tensor &input,
                       const Tensor &weight,
                       const Tensor &weight_scale,
                       const std::optional<Tensor> &bias,
                       Tensor q_input,
                       Tensor input_scale,
                       const Tensor &smooth_scale,
                       const char *op_name) {
    check_tensors(input, weight, weight_scale, bias, q_input, input_scale, smooth_scale, op_name);
    check_output(output, input, weight, op_name);
    check_workspace_devices(output, weight_scale, bias, q_input, input_scale, smooth_scale, op_name);
}

void lightop_per_token_dynamic_quant_int8(const Tensor &q_input,
                                          const Tensor &input,
                                          const Tensor &input_scale,
                                          const Tensor &smooth_scale) {
    auto q_input_at = infinicore::adaptor::to_aten_tensor(q_input);
    auto input_at = infinicore::adaptor::to_aten_tensor(input);
    auto input_scale_at = infinicore::adaptor::to_aten_tensor(input_scale);
    auto smooth_scale_at = infinicore::adaptor::to_aten_tensor(smooth_scale);
    lightop_linear_symbols().per_token_dynamic_quant_int8(q_input_at, input_at, input_scale_at, smooth_scale_at);
}

void run_native_impl(Tensor output,
                     const Tensor &input,
                     const Tensor &weight,
                     const Tensor &weight_scale,
                     const std::optional<Tensor> &bias,
                     Tensor q_input,
                     Tensor input_scale,
                     const Tensor &smooth_scale) {
    constexpr const char *op_name = "deepseek_v4_lightop_linear_w8a8_smooth_";
    check_all_tensors(output, input, weight, weight_scale, bias, q_input, input_scale, smooth_scale, op_name);

    if (context::getDevice() != output->device()) {
        context::setDevice(output->device());
    }

#if defined(ENABLE_HYGON_API)
    const int device_index = static_cast<int>(output->device().getIndex());
    hipError_t set_device_status = hipSetDevice(device_index);
    if (set_device_status != hipSuccess) {
        throw std::runtime_error(std::string(op_name) + " failed to set HIP device " +
                                 std::to_string(device_index) + ": " + hipGetErrorString(set_device_status));
    }
    c10::hip::HIPGuard device_guard(static_cast<c10::DeviceIndex>(device_index));
    c10::hip::HIPStreamGuard stream_guard(infinicore::adaptor::get_hip_stream());
#elif defined(ENABLE_NVIDIA_API)
    const int device_index = static_cast<int>(output->device().getIndex());
    c10::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(device_index));
    c10::cuda::CUDAStreamGuard stream_guard(infinicore::adaptor::get_cuda_stream());
#endif

    lightop_per_token_dynamic_quant_int8(q_input, input, input_scale, smooth_scale);
    compute_native_from_quantized(output, q_input, weight, input_scale, weight_scale, bias);
}

void run_impl(Tensor output,
              const Tensor &input,
              const Tensor &weight,
              const Tensor &weight_scale,
              const std::optional<Tensor> &bias,
              Tensor q_input,
              Tensor input_scale,
              const Tensor &smooth_scale,
              bool is_tuned_slide_block,
              int tuned_slide_block,
              bool allow_lightop_asm) {
    constexpr const char *op_name = "deepseek_v4_lightop_linear_w8a8_smooth_";
    check_all_tensors(output, input, weight, weight_scale, bias, q_input, input_scale, smooth_scale, op_name);

    if (!allow_lightop_asm || disable_lightop_asm()) {
#if defined(ENABLE_HYGON_API)
        if (debug_enabled() && allow_lightop_asm && disable_lightop_asm()) {
            std::fprintf(stderr,
                         "[deepseek_v4_lightop_linear_w8a8_smooth_] force native path by INFINICORE_LIGHTOP_LINEAR_DISABLE_ASM\n");
        }
#endif
        run_native_impl(output, input, weight, weight_scale, bias, q_input, input_scale, smooth_scale);
        return;
    }

    if (context::getDevice() != output->device()) {
        context::setDevice(output->device());
    }

#if defined(ENABLE_HYGON_API)
    const int device_index = static_cast<int>(output->device().getIndex());
    int hip_device_before = -1;
    (void)hipGetDevice(&hip_device_before);
    hipError_t set_device_status = hipSetDevice(device_index);
    if (set_device_status != hipSuccess) {
        throw std::runtime_error(std::string(op_name) + " failed to set HIP device " +
                                 std::to_string(device_index) + ": " + hipGetErrorString(set_device_status));
    }
#elif defined(ENABLE_NVIDIA_API)
    const int device_index = static_cast<int>(output->device().getIndex());
#endif

    {
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPGuard device_guard(static_cast<c10::DeviceIndex>(device_index));
    c10::hip::HIPStreamGuard stream_guard(infinicore::adaptor::get_hip_stream());
#elif defined(ENABLE_NVIDIA_API)
    c10::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(device_index));
    c10::cuda::CUDAStreamGuard stream_guard(infinicore::adaptor::get_cuda_stream());
#endif

#if defined(ENABLE_HYGON_API)
    if (debug_enabled()) {
        int hip_device_after = -1;
        (void)hipGetDevice(&hip_device_after);
        std::fprintf(stderr,
                     "[deepseek_v4_lightop_linear_w8a8_smooth_] context_device=%zu output_device=%zu hip_before=%d hip_after=%d stream=%p M=%zu N=%zu K=%zu allow_lightop_asm=%d\n",
                     context::getDevice().getIndex(),
                     output->device().getIndex(),
                     hip_device_before,
                     hip_device_after,
                     reinterpret_cast<void *>(context::getStream()),
                     input->size(0),
                     weight->size(0),
                     input->size(1),
                     allow_lightop_asm ? 1 : 0);
    }
#endif

    lightop_per_token_dynamic_quant_int8(q_input, input, input_scale, smooth_scale);
    debug_check_hip("after_quant", output);

    auto q_input_at = infinicore::adaptor::to_aten_tensor(q_input);
    auto input_scale_at = infinicore::adaptor::to_aten_tensor(input_scale);
    auto weight_scale_at = infinicore::adaptor::to_aten_tensor(weight_scale);
    auto output_at = infinicore::adaptor::to_aten_tensor(output);
    auto weight_raw_at = infinicore::adaptor::to_aten_tensor(weight);
    std::optional<at::Tensor> bias_at = std::nullopt;
    if (bias.has_value()) {
        bias_at = infinicore::adaptor::to_aten_tensor(*bias);
    }

    if (can_use_lightop_asm(q_input->size(0), q_input->size(1), bias_at)) {
        const auto &symbols = lightop_linear_symbols();
        auto weight_at = weight_raw_at.transpose(0, 1);
        at::Tensor result_at;
        if (debug_enabled()) {
            std::fprintf(stderr,
                         "[deepseek_v4_lightop_linear_w8a8_smooth_] enter_asm M=%zu N=%zu K=%zu is_tuned=%d tuned_slide_block=%d\n",
                         q_input->size(0),
                         weight->size(0),
                         q_input->size(1),
                         is_tuned_slide_block ? 1 : 0,
                         tuned_slide_block);
        }
        {
            std::lock_guard<std::mutex> lock(lightop_call_mutex());
            result_at = symbols.gemm_w8a8_smooth_asm(q_input_at,
                                                      weight_at,
                                                      input_scale_at,
                                                      weight_scale_at,
                                                      std::nullopt,
                                                      at::kBFloat16,
                                                      static_cast<int>(q_input->size(0)),
                                                      static_cast<int>(weight->size(0)),
                                                      static_cast<int>(q_input->size(1)),
                                                      is_tuned_slide_block,
                                                      tuned_slide_block);
        }
        debug_check_hip("after_asm", output);
        if (result_at.dim() != 2 || result_at.size(0) != output_at.size(0) ||
            result_at.size(1) != output_at.size(1) || result_at.device() != output_at.device()) {
            throw std::runtime_error(std::string(op_name) + " lightop result shape or device mismatch.");
        }
        output_at.copy_(result_at);
        debug_check_hip("after_copy", output);
    } else {
        compute_native_from_quantized(output, q_input, weight, input_scale, weight_scale, bias);
        debug_check_hip("after_native_gemm", output);
    }
    }

#if defined(ENABLE_HYGON_API)
    hipError_t leave_device_status = hipSetDevice(device_index);
    if (leave_device_status != hipSuccess) {
        throw std::runtime_error(std::string(op_name) + " failed to leave HIP device " +
                                 std::to_string(device_index) + " current: " +
                                 hipGetErrorString(leave_device_status));
    }
#endif
}
#endif

} // namespace

DeepseekV4LightopLinearW8A8Smooth::DeepseekV4LightopLinearW8A8Smooth(Tensor output,
                                                                     const Tensor &input,
                                                                     const Tensor &weight,
                                                                     const Tensor &weight_scale,
                                                                     const std::optional<Tensor> &bias,
                                                                     Tensor q_input,
                                                                     Tensor input_scale,
                                                                     const Tensor &smooth_scale,
                                                                     bool is_tuned_slide_block,
                                                                     int tuned_slide_block) {
    // The eager path may use lightop's ATen bridge, and the graph path currently
    // falls back to a native reference GEMM. Keep it in InfiniCore graph op-list
    // replay, but do not let device graph capture include this bridge/fallback
    // until lightop exposes a true out-workspace low-level GEMM entry.
    device_graph_capture_supported_ = false;
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(),
                                 output,
                                 input,
                                 weight,
                                 weight_scale,
                                 bias,
                                 q_input,
                                 input_scale,
                                 smooth_scale,
                                 is_tuned_slide_block,
                                 tuned_slide_block);
}

void DeepseekV4LightopLinearW8A8Smooth::execute(Tensor output,
                                                const Tensor &input,
                                                const Tensor &weight,
                                                const Tensor &weight_scale,
                                                const std::optional<Tensor> &bias,
                                                Tensor q_input,
                                                Tensor input_scale,
                                                const Tensor &smooth_scale,
                                                bool is_tuned_slide_block,
                                                int tuned_slide_block) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4LightopLinearW8A8Smooth,
                                      output,
                                      input,
                                      weight,
                                      weight_scale,
                                      bias,
                                      q_input,
                                      input_scale,
                                      smooth_scale,
                                      is_tuned_slide_block,
                                      tuned_slide_block);
}

namespace deepseek_v4_lightop_linear_w8a8_smooth_graph_impl {

struct PlannedMeta {
    graph::GraphTensor output;
    graph::GraphTensor input;
    graph::GraphTensor weight;
    graph::GraphTensor weight_scale;
    std::optional<graph::GraphTensor> bias;
    graph::GraphTensor q_input;
    graph::GraphTensor input_scale;
    graph::GraphTensor smooth_scale;
    bool is_tuned_slide_block;
    int tuned_slide_block;
};

std::optional<graph::GraphTensor> to_graph_optional(const std::optional<Tensor> &tensor) {
    if (tensor.has_value()) {
        return graph::GraphTensor(*tensor);
    }
    return std::nullopt;
}

std::optional<Tensor> to_tensor_optional(const std::optional<graph::GraphTensor> &tensor) {
    if (tensor.has_value()) {
        return *tensor;
    }
    return std::nullopt;
}

void *plan(Tensor output,
           const Tensor &input,
           const Tensor &weight,
           const Tensor &weight_scale,
           const std::optional<Tensor> &bias,
           Tensor q_input,
           Tensor input_scale,
           const Tensor &smooth_scale,
           bool is_tuned_slide_block,
           int tuned_slide_block) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_all_tensors(output,
                      input,
                      weight,
                      weight_scale,
                      bias,
                      q_input,
                      input_scale,
                      smooth_scale,
                      "deepseek_v4_lightop_linear_w8a8_smooth_");
#else
    (void)output;
    (void)input;
    (void)weight;
    (void)weight_scale;
    (void)bias;
    (void)q_input;
    (void)input_scale;
    (void)smooth_scale;
#endif
    return new PlannedMeta{graph::GraphTensor(output),
                           graph::GraphTensor(input),
                           graph::GraphTensor(weight),
                           graph::GraphTensor(weight_scale),
                           to_graph_optional(bias),
                           graph::GraphTensor(q_input),
                           graph::GraphTensor(input_scale),
                           graph::GraphTensor(smooth_scale),
                           is_tuned_slide_block,
                           tuned_slide_block};
}

void run(void *planned_meta) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    run_impl(planned->output,
             planned->input,
             planned->weight,
             planned->weight_scale,
             to_tensor_optional(planned->bias),
             planned->q_input,
             planned->input_scale,
             planned->smooth_scale,
             planned->is_tuned_slide_block,
             planned->tuned_slide_block,
             true);
#else
    (void)planned_meta;
    throw std::runtime_error("deepseek_v4_lightop_linear_w8a8_smooth_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_lightop_linear_w8a8_smooth_graph_impl

namespace deepseek_v4_lightop_linear_w8a8_smooth_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4LightopLinearW8A8Smooth,
                                       &deepseek_v4_lightop_linear_w8a8_smooth_graph_impl::plan,
                                       &deepseek_v4_lightop_linear_w8a8_smooth_graph_impl::run,
                                       &deepseek_v4_lightop_linear_w8a8_smooth_graph_impl::cleanup);
} // namespace deepseek_v4_lightop_linear_w8a8_smooth_register

void deepseek_v4_lightop_linear_w8a8_smooth_(Tensor output,
                                             const Tensor &input,
                                             const Tensor &weight,
                                             const Tensor &weight_scale,
                                             const std::optional<Tensor> &bias,
                                             Tensor q_input,
                                             Tensor input_scale,
                                             const Tensor &smooth_scale,
                                             bool is_tuned_slide_block,
                                             int tuned_slide_block) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    DeepseekV4LightopLinearW8A8Smooth::execute(output,
                                               input,
                                               weight,
                                               weight_scale,
                                               bias,
                                               q_input,
                                               input_scale,
                                               smooth_scale,
                                               is_tuned_slide_block,
                                               tuned_slide_block);
#else
    (void)output;
    (void)input;
    (void)weight;
    (void)weight_scale;
    (void)bias;
    (void)q_input;
    (void)input_scale;
    (void)smooth_scale;
    (void)is_tuned_slide_block;
    (void)tuned_slide_block;
    throw std::runtime_error("deepseek_v4_lightop_linear_w8a8_smooth_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void deepseek_v4_lightop_linear_w8a8_smooth_(Tensor output,
                                             const Tensor &input,
                                             const Tensor &weight,
                                             const Tensor &weight_scale,
                                             const std::optional<Tensor> &bias,
                                             Tensor q_input,
                                             Tensor input_scale,
                                             const Tensor &smooth_scale) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    const auto config = select_smooth_asm_config(static_cast<int64_t>(input->size(0)),
                                                 static_cast<int64_t>(weight->size(0)),
                                                 static_cast<int64_t>(input->size(1)));
    deepseek_v4_lightop_linear_w8a8_smooth_(output,
                                            input,
                                            weight,
                                            weight_scale,
                                            bias,
                                            q_input,
                                            input_scale,
                                            smooth_scale,
                                            config.first,
                                            config.second);
#else
    (void)output;
    (void)input;
    (void)weight;
    (void)weight_scale;
    (void)bias;
    (void)q_input;
    (void)input_scale;
    (void)smooth_scale;
    throw std::runtime_error("deepseek_v4_lightop_linear_w8a8_smooth_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
