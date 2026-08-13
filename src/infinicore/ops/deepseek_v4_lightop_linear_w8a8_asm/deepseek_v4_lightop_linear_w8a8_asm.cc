#include "infinicore/ops/deepseek_v4_lightop_linear_w8a8_asm.hpp"

#include "deepseek_v4_lightop_linear_w8a8_asm_kernel.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"

#ifdef ENABLE_ATEN
#include <ATen/ATen.h>
#include <ATen/core/ScalarType.h>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>
#elif defined(ENABLE_NVIDIA_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4LightopLinearW8A8Asm);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4LightopLinearW8A8AsmPerChannel);

namespace {

#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
constexpr size_t kBlockSize = 128;

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
    return "gfx936";
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

struct LightopW8A8AsmSymbols {
    using GemmW8A8AsmFn = at::Tensor (*)(at::Tensor &,
                                         at::Tensor &,
                                         at::Tensor &,
                                         at::Tensor &,
                                         std::optional<at::Tensor> &,
                                         std::optional<at::Tensor> &,
                                         std::optional<at::Tensor>,
                                         c10::ScalarType,
                                         int,
                                         int,
                                         int,
                                         int);
    using PerTokenDynamicQuantInt8Fn = void (*)(at::Tensor &, const at::Tensor &, at::Tensor &, const at::Tensor &);

    void *handle{nullptr};
    GemmW8A8AsmFn gemm_w8a8_asm{nullptr};
    PerTokenDynamicQuantInt8Fn per_token_dynamic_quant_int8{nullptr};
};

const LightopW8A8AsmSymbols &lightop_w8a8_asm_symbols() {
    static LightopW8A8AsmSymbols symbols;
    static std::once_flag once;
    std::call_once(once, [] {
        symbols.handle = open_lightop_so();
        symbols.gemm_w8a8_asm = checked_symbol<LightopW8A8AsmSymbols::GemmW8A8AsmFn>(
            symbols.handle,
            "_ZN2at6native13gemm_w8a8_asmERNS_6TensorES2_S2_S2_RSt8optionalIS1_ES5_S4_N3c1010ScalarTypeEiiii");
        symbols.per_token_dynamic_quant_int8 =
            checked_symbol<LightopW8A8AsmSymbols::PerTokenDynamicQuantInt8Fn>(
                symbols.handle,
                "_ZN2at6native28per_token_dynamic_quant_int8ERNS_6TensorERKS1_S2_S4_");
    });
    return symbols;
}

std::recursive_mutex &lightop_call_mutex() {
    static std::recursive_mutex mutex;
    return mutex;
}

bool env_enabled(const char *name) {
    const char *value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool disable_lightop_asm() {
    static const bool enabled = env_enabled("INFINICORE_LIGHTOP_LINEAR_W8A8_ASM_DISABLE");
    return enabled;
}

bool can_use_lightop_asm(int64_t m, int64_t n, int64_t k) {
    (void)n;
    (void)k;
    // The lightop asm GEMM is tuned for non-trivial M. Keep tiny decode rows on
    // the explicit fallback and use the SO only when it can amortize launch cost.
    return m > 16;
}

size_t ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

void check_tensors(Tensor output,
                   const Tensor &q_input,
                   const Tensor &weight,
                   const Tensor &input_block_scale,
                   const Tensor &weight_block_scale,
                   const char *op_name) {
    guard_device(output, op_name);
    guard_device(q_input, op_name);
    guard_device(weight, op_name);
    guard_device(input_block_scale, op_name);
    guard_device(weight_block_scale, op_name);

    if (q_input->ndim() != 2 || weight->ndim() != 2 || output->ndim() != 2 ||
        input_block_scale->ndim() != 2 || weight_block_scale->ndim() != 2) {
        throw std::runtime_error(std::string(op_name) + " expects 2D tensors.");
    }
    if (q_input->dtype() != DataType::I8 || weight->dtype() != DataType::I8 ||
        input_block_scale->dtype() != DataType::F32 || weight_block_scale->dtype() != DataType::F32 ||
        output->dtype() != DataType::BF16) {
        throw std::runtime_error(std::string(op_name) + " expects q_input/weight I8, scales F32, output BF16.");
    }
    if (!q_input->is_contiguous() || !weight->is_contiguous() || !input_block_scale->is_contiguous() ||
        !weight_block_scale->is_contiguous() || !output->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous tensors.");
    }
    if (output->device() != q_input->device() || output->device() != weight->device() ||
        output->device() != input_block_scale->device() || output->device() != weight_block_scale->device()) {
        throw std::runtime_error(std::string(op_name) + " expects all tensors on the same device.");
    }

    const size_t m = q_input->size(0);
    const size_t k = q_input->size(1);
    const size_t n = weight->size(0);
    const size_t k_blocks = ceil_div(k, kBlockSize);
    const size_t n_blocks = ceil_div(n, kBlockSize);
    if (weight->size(1) != k || output->shape() != std::vector<size_t>{m, n}) {
        throw std::runtime_error(std::string(op_name) + " output or weight shape mismatch.");
    }
    if (k % kBlockSize != 0 || n % kBlockSize != 0) {
        throw std::runtime_error(std::string(op_name) + " requires N and K divisible by 128.");
    }
    if (input_block_scale->shape() != std::vector<size_t>{k_blocks, m}) {
        throw std::runtime_error(std::string(op_name) + " expects input_block_scale [ceil(K/128), M].");
    }
    if (weight_block_scale->shape() != std::vector<size_t>{n_blocks, k_blocks}) {
        throw std::runtime_error(std::string(op_name) + " expects weight_block_scale [ceil(N/128), ceil(K/128)].");
    }
}

void compute_reference(Tensor output,
                       const Tensor &q_input,
                       const Tensor &weight,
                       const Tensor &input_block_scale,
                       const Tensor &weight_block_scale) {
    auto output_at = infinicore::adaptor::to_aten_tensor(output);
    auto q_at = infinicore::adaptor::to_aten_tensor(q_input);
    auto w_at = infinicore::adaptor::to_aten_tensor(weight);
    auto a_scale_at = infinicore::adaptor::to_aten_tensor(input_block_scale);
    auto b_scale_at = infinicore::adaptor::to_aten_tensor(weight_block_scale);

    const int64_t m = q_at.size(0);
    const int64_t n = w_at.size(0);
    const int64_t k = q_at.size(1);
    const int64_t k_blocks = (k + static_cast<int64_t>(kBlockSize) - 1) / static_cast<int64_t>(kBlockSize);
    const int64_t n_blocks = (n + static_cast<int64_t>(kBlockSize) - 1) / static_cast<int64_t>(kBlockSize);

    auto acc = at::zeros({m, n}, q_at.options().dtype(at::kFloat));
    for (int64_t kb = 0; kb < k_blocks; ++kb) {
        const int64_t k_start = kb * static_cast<int64_t>(kBlockSize);
        auto partial = at::matmul(q_at.narrow(1, k_start, static_cast<int64_t>(kBlockSize)).to(at::kFloat),
                                  w_at.narrow(1, k_start, static_cast<int64_t>(kBlockSize)).to(at::kFloat).transpose(0, 1));
        auto row_scale = a_scale_at.select(0, kb).reshape({m, 1});
        auto col_scale = b_scale_at.select(1, kb)
                             .unsqueeze(1)
                             .repeat({1, static_cast<int64_t>(kBlockSize)})
                             .reshape({n_blocks * static_cast<int64_t>(kBlockSize)})
                             .slice(0, 0, n)
                             .reshape({1, n});
        partial.mul_(row_scale);
        partial.mul_(col_scale);
        acc.add_(partial);
    }
    output_at.copy_(acc.to(output_at.scalar_type()));
}

void lightop_per_token_dynamic_quant_int8(const Tensor &q_input,
                                          const Tensor &input,
                                          const Tensor &input_scale,
                                          const Tensor &smooth_scale) {
    auto q_input_at = infinicore::adaptor::to_aten_tensor(q_input);
    auto input_at = infinicore::adaptor::to_aten_tensor(input);
    auto input_scale_at = infinicore::adaptor::to_aten_tensor(input_scale);
    auto smooth_scale_at = infinicore::adaptor::to_aten_tensor(smooth_scale);
    lightop_w8a8_asm_symbols().per_token_dynamic_quant_int8(q_input_at, input_at, input_scale_at, smooth_scale_at);
}

void native_per_token_quant_int8_bf16(const Tensor &q_input,
                                      const Tensor &input,
                                      const Tensor &input_scale,
                                      const Tensor &smooth_scale) {
    deepseek_v4_lightop_linear_w8a8_asm_impl::launch_per_token_quant_int8_bf16(
        reinterpret_cast<int8_t *>(const_cast<std::byte *>(q_input->data())),
        reinterpret_cast<float *>(const_cast<std::byte *>(input_scale->data())),
        input->data(),
        reinterpret_cast<const float *>(smooth_scale->data()),
        static_cast<int64_t>(input->size(0)),
        static_cast<int64_t>(input->size(1)),
        context::getStream());
}

void check_per_channel_tensors(Tensor output,
                               const Tensor &input,
                               const Tensor &weight,
                               const Tensor &weight_scale,
                               Tensor q_input,
                               Tensor input_scale,
                               Tensor input_block_scale,
                               Tensor weight_block_scale,
                               const Tensor &smooth_scale,
                               const char *op_name) {
    guard_device(output, op_name);
    guard_device(input, op_name);
    guard_device(weight, op_name);
    guard_device(weight_scale, op_name);
    guard_device(q_input, op_name);
    guard_device(input_scale, op_name);
    guard_device(input_block_scale, op_name);
    guard_device(weight_block_scale, op_name);
    guard_device(smooth_scale, op_name);

    if (input->ndim() != 2 || weight->ndim() != 2 || weight_scale->ndim() != 2 ||
        q_input->ndim() != 2 || input_scale->ndim() != 2 || input_block_scale->ndim() != 2 ||
        weight_block_scale->ndim() != 2 || smooth_scale->ndim() != 1 || output->ndim() != 2) {
        throw std::runtime_error(std::string(op_name) + " expects 2D tensors and 1D smooth_scale.");
    }
    if (input->dtype() != DataType::BF16 || output->dtype() != DataType::BF16 ||
        weight->dtype() != DataType::I8 || q_input->dtype() != DataType::I8 ||
        weight_scale->dtype() != DataType::F32 || input_scale->dtype() != DataType::F32 ||
        input_block_scale->dtype() != DataType::F32 || weight_block_scale->dtype() != DataType::F32 ||
        smooth_scale->dtype() != DataType::F32) {
        throw std::runtime_error(std::string(op_name) + " unexpected dtype.");
    }
    if (!input->is_contiguous() || !output->is_contiguous() || !weight->is_contiguous() ||
        !weight_scale->is_contiguous() || !q_input->is_contiguous() || !input_scale->is_contiguous() ||
        !input_block_scale->is_contiguous() || !weight_block_scale->is_contiguous() || !smooth_scale->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous tensors.");
    }
    if (output->device() != input->device() || output->device() != weight->device() ||
        output->device() != weight_scale->device() || output->device() != q_input->device() ||
        output->device() != input_scale->device() || output->device() != input_block_scale->device() ||
        output->device() != weight_block_scale->device() || output->device() != smooth_scale->device()) {
        throw std::runtime_error(std::string(op_name) + " expects all tensors on the same device.");
    }

    const size_t m = input->size(0);
    const size_t k = input->size(1);
    const size_t n = weight->size(0);
    const size_t k_blocks = ceil_div(k, kBlockSize);
    const size_t n_blocks = ceil_div(n, kBlockSize);
    if (k % kBlockSize != 0 || n % kBlockSize != 0) {
        throw std::runtime_error(std::string(op_name) + " requires N and K divisible by 128.");
    }
    if (weight->size(1) != k || output->shape() != std::vector<size_t>{m, n} ||
        q_input->shape() != std::vector<size_t>{m, k} ||
        input_scale->shape() != std::vector<size_t>{m, 1} ||
        input_block_scale->shape() != std::vector<size_t>{k_blocks, m} ||
        weight_block_scale->shape() != std::vector<size_t>{n_blocks, k_blocks} ||
        weight_scale->shape() != std::vector<size_t>{n, 1} ||
        smooth_scale->shape() != std::vector<size_t>{k}) {
        throw std::runtime_error(std::string(op_name) + " shape mismatch.");
    }
}

void run_impl(Tensor output,
              const Tensor &q_input,
              const Tensor &weight,
              const Tensor &input_block_scale,
              const Tensor &weight_block_scale,
              bool allow_lightop_asm) {
    constexpr const char *op_name = "deepseek_v4_lightop_linear_w8a8_asm_";
    check_tensors(output, q_input, weight, input_block_scale, weight_block_scale, op_name);

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

    if (!allow_lightop_asm || disable_lightop_asm() ||
        !can_use_lightop_asm(static_cast<int64_t>(q_input->size(0)),
                             static_cast<int64_t>(weight->size(0)),
                             static_cast<int64_t>(q_input->size(1)))) {
        compute_reference(output, q_input, weight, input_block_scale, weight_block_scale);
        return;
    }

    auto output_at = infinicore::adaptor::to_aten_tensor(output);
    auto q_at = infinicore::adaptor::to_aten_tensor(q_input);
    auto w_at = infinicore::adaptor::to_aten_tensor(weight);
    auto a_scale_at = infinicore::adaptor::to_aten_tensor(input_block_scale);
    auto b_scale_at = infinicore::adaptor::to_aten_tensor(weight_block_scale);

    std::optional<at::Tensor> c1 = output_at;
    std::optional<at::Tensor> c2 = output_at;
    const auto &symbols = lightop_w8a8_asm_symbols();
    at::Tensor result_at;
    {
        std::lock_guard<std::recursive_mutex> lock(lightop_call_mutex());
        result_at = symbols.gemm_w8a8_asm(q_at,
                                           w_at,
                                           a_scale_at,
                                           b_scale_at,
                                           c1,
                                           c2,
                                           std::nullopt,
                                           at::kBFloat16,
                                           static_cast<int>(q_input->size(0)),
                                           static_cast<int>(weight->size(0)),
                                           static_cast<int>(q_input->size(1)),
                                           0);
    }
    if (result_at.dim() != 2 || result_at.size(0) != output_at.size(0) ||
        result_at.size(1) != output_at.size(1) || result_at.device() != output_at.device() ||
        result_at.data_ptr() != output_at.data_ptr()) {
        throw std::runtime_error(std::string(op_name) + " lightop output tensor mismatch.");
    }
}

void run_per_channel_impl(Tensor output,
                          const Tensor &input,
                          const Tensor &weight,
                          const Tensor &weight_scale,
                          Tensor q_input,
                          Tensor input_scale,
                          Tensor input_block_scale,
                          Tensor weight_block_scale,
                          const Tensor &smooth_scale,
                          bool allow_lightop_asm) {
    constexpr const char *op_name = "deepseek_v4_lightop_linear_w8a8_asm_per_channel_";
    check_per_channel_tensors(output,
                              input,
                              weight,
                              weight_scale,
                              q_input,
                              input_scale,
                              input_block_scale,
                              weight_block_scale,
                              smooth_scale,
                              op_name);

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

    const bool use_lightop = allow_lightop_asm && !disable_lightop_asm() &&
                             can_use_lightop_asm(static_cast<int64_t>(input->size(0)),
                                                 static_cast<int64_t>(weight->size(0)),
                                                 static_cast<int64_t>(input->size(1)));
    if (use_lightop) {
        std::lock_guard<std::recursive_mutex> lock(lightop_call_mutex());
        lightop_per_token_dynamic_quant_int8(q_input, input, input_scale, smooth_scale);
        deepseek_v4_lightop_linear_w8a8_asm_impl::launch_prepare_per_channel_scales(
            reinterpret_cast<float *>(input_block_scale->data()),
            reinterpret_cast<float *>(weight_block_scale->data()),
            reinterpret_cast<const float *>(input_scale->data()),
            static_cast<int64_t>(input->size(0)),
            static_cast<int64_t>(weight->size(0)),
            static_cast<int64_t>(input->size(1)),
            context::getStream());

        run_impl(output, q_input, weight, input_block_scale, weight_block_scale, true);

        deepseek_v4_lightop_linear_w8a8_asm_impl::launch_apply_per_channel_weight_scale(
            output->data(),
            reinterpret_cast<const float *>(weight_scale->data()),
            static_cast<int64_t>(output->size(0)),
            static_cast<int64_t>(output->size(1)),
            context::getStream());
        return;
    } else {
        native_per_token_quant_int8_bf16(q_input, input, input_scale, smooth_scale);
    }
    deepseek_v4_lightop_linear_w8a8_asm_impl::launch_prepare_per_channel_scales(
        reinterpret_cast<float *>(input_block_scale->data()),
        reinterpret_cast<float *>(weight_block_scale->data()),
        reinterpret_cast<const float *>(input_scale->data()),
        static_cast<int64_t>(input->size(0)),
        static_cast<int64_t>(weight->size(0)),
        static_cast<int64_t>(input->size(1)),
        context::getStream());

    run_impl(output, q_input, weight, input_block_scale, weight_block_scale, use_lightop);

    deepseek_v4_lightop_linear_w8a8_asm_impl::launch_apply_per_channel_weight_scale(
        output->data(),
        reinterpret_cast<const float *>(weight_scale->data()),
        static_cast<int64_t>(output->size(0)),
        static_cast<int64_t>(output->size(1)),
        context::getStream());
}
#endif

} // namespace

DeepseekV4LightopLinearW8A8Asm::DeepseekV4LightopLinearW8A8Asm(Tensor output,
                                                               const Tensor &q_input,
                                                               const Tensor &weight,
                                                               const Tensor &input_block_scale,
                                                               const Tensor &weight_block_scale) {
    device_graph_capture_supported_ = false;
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(),
                                 output,
                                 q_input,
                                 weight,
                                 input_block_scale,
                                 weight_block_scale);
}

void DeepseekV4LightopLinearW8A8Asm::execute(Tensor output,
                                             const Tensor &q_input,
                                             const Tensor &weight,
                                             const Tensor &input_block_scale,
                                             const Tensor &weight_block_scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4LightopLinearW8A8Asm,
                                      output,
                                      q_input,
                                      weight,
                                      input_block_scale,
                                      weight_block_scale);
}

namespace deepseek_v4_lightop_linear_w8a8_asm_graph_impl {

struct PlannedMeta {
    graph::GraphTensor output;
    graph::GraphTensor q_input;
    graph::GraphTensor weight;
    graph::GraphTensor input_block_scale;
    graph::GraphTensor weight_block_scale;
};

void *plan(Tensor output,
           const Tensor &q_input,
           const Tensor &weight,
           const Tensor &input_block_scale,
           const Tensor &weight_block_scale) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_tensors(output,
                  q_input,
                  weight,
                  input_block_scale,
                  weight_block_scale,
                  "deepseek_v4_lightop_linear_w8a8_asm_");
#else
    (void)output;
    (void)q_input;
    (void)weight;
    (void)input_block_scale;
    (void)weight_block_scale;
#endif
    return new PlannedMeta{graph::GraphTensor(output),
                           graph::GraphTensor(q_input),
                           graph::GraphTensor(weight),
                           graph::GraphTensor(input_block_scale),
                           graph::GraphTensor(weight_block_scale)};
}

void run(void *planned_meta) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    run_impl(planned->output,
             planned->q_input,
             planned->weight,
             planned->input_block_scale,
             planned->weight_block_scale,
             false);
#else
    (void)planned_meta;
    throw std::runtime_error("deepseek_v4_lightop_linear_w8a8_asm_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_lightop_linear_w8a8_asm_graph_impl

namespace deepseek_v4_lightop_linear_w8a8_asm_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4LightopLinearW8A8Asm,
                                       &deepseek_v4_lightop_linear_w8a8_asm_graph_impl::plan,
                                       &deepseek_v4_lightop_linear_w8a8_asm_graph_impl::run,
                                       &deepseek_v4_lightop_linear_w8a8_asm_graph_impl::cleanup);
} // namespace deepseek_v4_lightop_linear_w8a8_asm_register

DeepseekV4LightopLinearW8A8AsmPerChannel::DeepseekV4LightopLinearW8A8AsmPerChannel(Tensor output,
                                                                                   const Tensor &input,
                                                                                   const Tensor &weight,
                                                                                   const Tensor &weight_scale,
                                                                                   Tensor q_input,
                                                                                   Tensor input_scale,
                                                                                   Tensor input_block_scale,
                                                                                   Tensor weight_block_scale,
                                                                                   const Tensor &smooth_scale) {
    device_graph_capture_supported_ = false;
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(),
                                 output,
                                 input,
                                 weight,
                                 weight_scale,
                                 q_input,
                                 input_scale,
                                 input_block_scale,
                                 weight_block_scale,
                                 smooth_scale);
}

void DeepseekV4LightopLinearW8A8AsmPerChannel::execute(Tensor output,
                                                       const Tensor &input,
                                                       const Tensor &weight,
                                                       const Tensor &weight_scale,
                                                       Tensor q_input,
                                                       Tensor input_scale,
                                                       Tensor input_block_scale,
                                                       Tensor weight_block_scale,
                                                       const Tensor &smooth_scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4LightopLinearW8A8AsmPerChannel,
                                      output,
                                      input,
                                      weight,
                                      weight_scale,
                                      q_input,
                                      input_scale,
                                      input_block_scale,
                                      weight_block_scale,
                                      smooth_scale);
}

namespace deepseek_v4_lightop_linear_w8a8_asm_per_channel_graph_impl {

struct PlannedMeta {
    graph::GraphTensor output;
    graph::GraphTensor input;
    graph::GraphTensor weight;
    graph::GraphTensor weight_scale;
    graph::GraphTensor q_input;
    graph::GraphTensor input_scale;
    graph::GraphTensor input_block_scale;
    graph::GraphTensor weight_block_scale;
    graph::GraphTensor smooth_scale;
};

void *plan(Tensor output,
           const Tensor &input,
           const Tensor &weight,
           const Tensor &weight_scale,
           Tensor q_input,
           Tensor input_scale,
           Tensor input_block_scale,
           Tensor weight_block_scale,
           const Tensor &smooth_scale) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_per_channel_tensors(output,
                              input,
                              weight,
                              weight_scale,
                              q_input,
                              input_scale,
                              input_block_scale,
                              weight_block_scale,
                              smooth_scale,
                              "deepseek_v4_lightop_linear_w8a8_asm_per_channel_");
#else
    (void)output;
    (void)input;
    (void)weight;
    (void)weight_scale;
    (void)q_input;
    (void)input_scale;
    (void)input_block_scale;
    (void)weight_block_scale;
    (void)smooth_scale;
#endif
    return new PlannedMeta{graph::GraphTensor(output),
                           graph::GraphTensor(input),
                           graph::GraphTensor(weight),
                           graph::GraphTensor(weight_scale),
                           graph::GraphTensor(q_input),
                           graph::GraphTensor(input_scale),
                           graph::GraphTensor(input_block_scale),
                           graph::GraphTensor(weight_block_scale),
                           graph::GraphTensor(smooth_scale)};
}

void run(void *planned_meta) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    run_per_channel_impl(planned->output,
                         planned->input,
                         planned->weight,
                         planned->weight_scale,
                         planned->q_input,
                         planned->input_scale,
                         planned->input_block_scale,
                         planned->weight_block_scale,
                         planned->smooth_scale,
                         false);
#else
    (void)planned_meta;
    throw std::runtime_error("deepseek_v4_lightop_linear_w8a8_asm_per_channel_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_lightop_linear_w8a8_asm_per_channel_graph_impl

namespace deepseek_v4_lightop_linear_w8a8_asm_per_channel_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4LightopLinearW8A8AsmPerChannel,
                                       &deepseek_v4_lightop_linear_w8a8_asm_per_channel_graph_impl::plan,
                                       &deepseek_v4_lightop_linear_w8a8_asm_per_channel_graph_impl::run,
                                       &deepseek_v4_lightop_linear_w8a8_asm_per_channel_graph_impl::cleanup);
} // namespace deepseek_v4_lightop_linear_w8a8_asm_per_channel_register

void deepseek_v4_lightop_linear_w8a8_asm_(Tensor output,
                                          const Tensor &q_input,
                                          const Tensor &weight,
                                          const Tensor &input_block_scale,
                                          const Tensor &weight_block_scale) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    if (context::isGraphRecording()) {
        DeepseekV4LightopLinearW8A8Asm::execute(output,
                                                q_input,
                                                weight,
                                                input_block_scale,
                                                weight_block_scale);
    } else {
        run_impl(output, q_input, weight, input_block_scale, weight_block_scale, true);
    }
#else
    (void)output;
    (void)q_input;
    (void)weight;
    (void)input_block_scale;
    (void)weight_block_scale;
    throw std::runtime_error("deepseek_v4_lightop_linear_w8a8_asm_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void deepseek_v4_lightop_linear_w8a8_asm_per_channel_(Tensor output,
                                                      const Tensor &input,
                                                      const Tensor &weight,
                                                      const Tensor &weight_scale,
                                                      Tensor q_input,
                                                      Tensor input_scale,
                                                      Tensor input_block_scale,
                                                      Tensor weight_block_scale,
                                                      const Tensor &smooth_scale) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    if (context::isGraphRecording()) {
        DeepseekV4LightopLinearW8A8AsmPerChannel::execute(output,
                                                          input,
                                                          weight,
                                                          weight_scale,
                                                          q_input,
                                                          input_scale,
                                                          input_block_scale,
                                                          weight_block_scale,
                                                          smooth_scale);
    } else {
        run_per_channel_impl(output,
                             input,
                             weight,
                             weight_scale,
                             q_input,
                             input_scale,
                             input_block_scale,
                             weight_block_scale,
                             smooth_scale,
                             true);
    }
#else
    (void)output;
    (void)input;
    (void)weight;
    (void)weight_scale;
    (void)q_input;
    (void)input_scale;
    (void)input_block_scale;
    (void)weight_block_scale;
    (void)smooth_scale;
    throw std::runtime_error("deepseek_v4_lightop_linear_w8a8_asm_per_channel_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
