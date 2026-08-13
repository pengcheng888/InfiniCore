#include "infinicore/ops/deepseek_v4_lightop_per_token_dynamic_quant_int8.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ATen.h>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>
#elif defined(ENABLE_NVIDIA_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4LightopPerTokenDynamicQuantInt8);

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

struct LightopQuantSymbols {
    using PerTokenDynamicQuantInt8Fn = void (*)(at::Tensor &, const at::Tensor &, at::Tensor &, const at::Tensor &);

    void *handle{nullptr};
    PerTokenDynamicQuantInt8Fn per_token_dynamic_quant_int8{nullptr};
};

const LightopQuantSymbols &lightop_quant_symbols() {
    static LightopQuantSymbols symbols;
    static std::once_flag once;
    std::call_once(once, [] {
        symbols.handle = open_lightop_so();
        symbols.per_token_dynamic_quant_int8 =
            checked_symbol<LightopQuantSymbols::PerTokenDynamicQuantInt8Fn>(
                symbols.handle,
                "_ZN2at6native28per_token_dynamic_quant_int8ERNS_6TensorERKS1_S2_S4_");
    });
    return symbols;
}

void check_tensors(Tensor q_input, const Tensor &input, Tensor input_scale, const Tensor &smooth_scale, const char *op_name) {
    guard_device(q_input, op_name);
    guard_device(input, op_name);
    guard_device(input_scale, op_name);
    guard_device(smooth_scale, op_name);
    if (q_input->device() != input->device() || input_scale->device() != input->device() ||
        smooth_scale->device() != input->device()) {
        throw std::runtime_error(std::string(op_name) + " expects all tensors on the same device.");
    }
    if (input->ndim() != 2 || q_input->ndim() != 2 || input_scale->ndim() != 2 || smooth_scale->ndim() != 1) {
        throw std::runtime_error(std::string(op_name) + " expects q_input/input 2D, input_scale 2D, smooth_scale 1D.");
    }
    if ((input->dtype() != DataType::BF16 && input->dtype() != DataType::F16 && input->dtype() != DataType::F32) ||
        q_input->dtype() != DataType::I8 || input_scale->dtype() != DataType::F32 ||
        smooth_scale->dtype() != DataType::F32) {
        throw std::runtime_error(std::string(op_name) + " unexpected dtype.");
    }
    if (!input->is_contiguous() || !q_input->is_contiguous() || !input_scale->is_contiguous() ||
        !smooth_scale->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous tensors.");
    }
    const size_t rows = input->size(0);
    const size_t cols = input->size(1);
    if (q_input->shape() != input->shape() || input_scale->shape() != std::vector<size_t>{rows, 1} ||
        smooth_scale->size(0) != cols) {
        throw std::runtime_error(std::string(op_name) + " shape mismatch.");
    }
}

void run_impl(Tensor q_input, const Tensor &input, Tensor input_scale, const Tensor &smooth_scale) {
    constexpr const char *op_name = "deepseek_v4_lightop_per_token_dynamic_quant_int8_";
    check_tensors(q_input, input, input_scale, smooth_scale, op_name);

    if (context::getDevice() != input->device()) {
        context::setDevice(input->device());
    }

#if defined(ENABLE_HYGON_API)
    const int device_index = static_cast<int>(input->device().getIndex());
    hipError_t set_device_status = hipSetDevice(device_index);
    if (set_device_status != hipSuccess) {
        throw std::runtime_error(std::string(op_name) + " failed to set HIP device " +
                                 std::to_string(device_index) + ": " + hipGetErrorString(set_device_status));
    }
    c10::hip::HIPGuard device_guard(static_cast<c10::DeviceIndex>(device_index));
    c10::hip::HIPStreamGuard stream_guard(infinicore::adaptor::get_hip_stream());
#elif defined(ENABLE_NVIDIA_API)
    const int device_index = static_cast<int>(input->device().getIndex());
    c10::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(device_index));
    c10::cuda::CUDAStreamGuard stream_guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto q_input_at = infinicore::adaptor::to_aten_tensor(q_input);
    auto input_at = infinicore::adaptor::to_aten_tensor(input);
    auto input_scale_at = infinicore::adaptor::to_aten_tensor(input_scale);
    auto smooth_scale_at = infinicore::adaptor::to_aten_tensor(smooth_scale);
    lightop_quant_symbols().per_token_dynamic_quant_int8(q_input_at, input_at, input_scale_at, smooth_scale_at);
}
#endif

} // namespace

DeepseekV4LightopPerTokenDynamicQuantInt8::DeepseekV4LightopPerTokenDynamicQuantInt8(
    Tensor q_input,
    const Tensor &input,
    Tensor input_scale,
    const Tensor &smooth_scale) {
    device_graph_capture_supported_ = false;
    INFINICORE_GRAPH_OP_DISPATCH(input->device().getType(), q_input, input, input_scale, smooth_scale);
}

void DeepseekV4LightopPerTokenDynamicQuantInt8::execute(Tensor q_input,
                                                        const Tensor &input,
                                                        Tensor input_scale,
                                                        const Tensor &smooth_scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4LightopPerTokenDynamicQuantInt8,
                                      q_input,
                                      input,
                                      input_scale,
                                      smooth_scale);
}

namespace deepseek_v4_lightop_per_token_dynamic_quant_int8_graph_impl {

struct PlannedMeta {
    graph::GraphTensor q_input;
    graph::GraphTensor input;
    graph::GraphTensor input_scale;
    graph::GraphTensor smooth_scale;
};

void *plan(Tensor q_input, const Tensor &input, Tensor input_scale, const Tensor &smooth_scale) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_tensors(q_input, input, input_scale, smooth_scale, "deepseek_v4_lightop_per_token_dynamic_quant_int8_");
#else
    (void)q_input;
    (void)input;
    (void)input_scale;
    (void)smooth_scale;
#endif
    return new PlannedMeta{graph::GraphTensor(q_input),
                           graph::GraphTensor(input),
                           graph::GraphTensor(input_scale),
                           graph::GraphTensor(smooth_scale)};
}

void run(void *planned_meta) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    run_impl(planned->q_input, planned->input, planned->input_scale, planned->smooth_scale);
#else
    (void)planned_meta;
    throw std::runtime_error("deepseek_v4_lightop_per_token_dynamic_quant_int8_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_lightop_per_token_dynamic_quant_int8_graph_impl

namespace deepseek_v4_lightop_per_token_dynamic_quant_int8_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4LightopPerTokenDynamicQuantInt8,
                                       &deepseek_v4_lightop_per_token_dynamic_quant_int8_graph_impl::plan,
                                       &deepseek_v4_lightop_per_token_dynamic_quant_int8_graph_impl::run,
                                       &deepseek_v4_lightop_per_token_dynamic_quant_int8_graph_impl::cleanup);
} // namespace deepseek_v4_lightop_per_token_dynamic_quant_int8_register

void deepseek_v4_lightop_per_token_dynamic_quant_int8_(Tensor q_input,
                                                       const Tensor &input,
                                                       Tensor input_scale,
                                                       const Tensor &smooth_scale) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    if (context::isGraphRecording()) {
        DeepseekV4LightopPerTokenDynamicQuantInt8::execute(q_input, input, input_scale, smooth_scale);
    } else {
        run_impl(q_input, input, input_scale, smooth_scale);
    }
#else
    (void)q_input;
    (void)input;
    (void)input_scale;
    (void)smooth_scale;
    throw std::runtime_error("deepseek_v4_lightop_per_token_dynamic_quant_int8_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
