#include "infinicore/ops/deepseek_v4_lmslim_rocblas_linear_w8a8.hpp"

#include "deepseek_v4_lmslim_rocblas_linear_w8a8_kernel.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/ops/deepseek_v4_lightop_per_token_dynamic_quant_int8.hpp"

#if defined(ENABLE_HYGON_API)
#include <hip/hip_runtime.h>
#include <rocblas.h>
#endif

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4LmslimRocblasLinearW8A8);

namespace {

#if defined(ENABLE_HYGON_API)
using RocblasInt8GemmFn = void (*)(hipStream_t, rocblas_handle, int *, const int8_t *, const int8_t *, int, int, int, bool);

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

struct LmslimRocblasSymbols {
    void *handle{nullptr};
    RocblasInt8GemmFn gemm_rocblas_int8_gpu{nullptr};
};

const LmslimRocblasSymbols &lmslim_rocblas_symbols() {
    static LmslimRocblasSymbols symbols;
    static std::once_flag once;
    std::call_once(once, [] {
        symbols.handle = open_lmslimquant_so();
        symbols.gemm_rocblas_int8_gpu = checked_symbol<RocblasInt8GemmFn>(
            symbols.handle,
            "_Z21gemm_rocblas_int8_gpuP12ihipStream_tP15_rocblas_handlePiPKaS5_iiib");
    });
    return symbols;
}

class RocblasHandlePool {
public:
    RocblasHandlePool() {
        handles_.fill(nullptr);
    }

    ~RocblasHandlePool() {
        for (auto *handle : handles_) {
            if (handle != nullptr) {
                (void)rocblas_destroy_handle(handle);
            }
        }
    }

    rocblas_handle get(size_t device_index) {
        if (device_index >= handles_.size()) {
            throw std::runtime_error("deepseek_v4_lmslim_rocblas_linear_w8a8_ supports at most 32 devices.");
        }
        std::lock_guard<std::mutex> guard(mutex_);
        auto *&handle = handles_[device_index];
        if (handle == nullptr) {
            const rocblas_status status = rocblas_create_handle(&handle);
            if (status != rocblas_status_success) {
                throw std::runtime_error("rocblas_create_handle failed for deepseek_v4_lmslim_rocblas_linear_w8a8_.");
            }
        }
        return handle;
    }

private:
    std::mutex mutex_;
    std::array<rocblas_handle, 32> handles_;
};

rocblas_handle rocblas_handle_for_device(size_t device_index) {
    static RocblasHandlePool pool;
    return pool.get(device_index);
}

void guard_device(const Tensor &tensor, const char *op_name) {
    if (tensor->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error(std::string(op_name) + " expects HYGON tensors in this build.");
    }
}
#endif

void check_same_device(const Tensor &base, const Tensor &other, const char *op_name, const char *arg_name) {
    if (base->device() != other->device()) {
        throw std::runtime_error(std::string(op_name) + " expects " + arg_name + " on the same device as output.");
    }
}

bool profile_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("INFINICORE_DSV4_LMSLIM_ROCBLAS_LINEAR_PROFILE");
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
                   Tensor accum,
                   const Tensor &smooth_scale,
                   const char *op_name) {
    if (output->ndim() != 2 || input->ndim() != 2 || weight_t->ndim() != 2 ||
        weight_scale->ndim() != 2 || q_input->ndim() != 2 || input_scale->ndim() != 2 ||
        accum->ndim() != 2 || smooth_scale->ndim() != 1) {
        throw std::runtime_error(std::string(op_name) + " expects 2D tensors and 1D smooth_scale.");
    }
    if (output->dtype() != input->dtype() ||
        (output->dtype() != DataType::BF16 && output->dtype() != DataType::F16) ||
        weight_t->dtype() != DataType::I8 || weight_scale->dtype() != DataType::F32 ||
        q_input->dtype() != DataType::I8 || input_scale->dtype() != DataType::F32 ||
        accum->dtype() != DataType::I32 || smooth_scale->dtype() != DataType::F32) {
        throw std::runtime_error(std::string(op_name) + " expects BF16/F16 output/input, I8 weights/q_input, I32 accum, and F32 scales.");
    }
    if (!output->is_contiguous() || !input->is_contiguous() ||
        !weight_scale->is_contiguous() || !q_input->is_contiguous() || !input_scale->is_contiguous() ||
        !accum->is_contiguous() || !smooth_scale->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous output/input/workspaces/scales.");
    }
    if (!weight_t->is_contiguous() && weight_t->stride(0) != 1) {
        throw std::runtime_error(std::string(op_name) + " expects weight_t [K,N] to be contiguous or a transpose view with stride(0)=1.");
    }

    check_same_device(output, input, op_name, "input");
    check_same_device(output, weight_t, op_name, "weight_t");
    check_same_device(output, weight_scale, op_name, "weight_scale");
    check_same_device(output, q_input, op_name, "q_input");
    check_same_device(output, input_scale, op_name, "input_scale");
    check_same_device(output, accum, op_name, "accum");
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
    if (q_input->shape() != input->shape() || input_scale->shape() != std::vector<size_t>{m, 1} ||
        accum->shape() != std::vector<size_t>{m, n}) {
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
    if (m > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        k > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(op_name) + " dimensions exceed lmslim rocBLAS int API range.");
    }
}

void run_impl(Tensor output,
              const Tensor &input,
              const Tensor &weight_t,
              const Tensor &weight_scale,
              std::optional<Tensor> bias,
              Tensor q_input,
              Tensor input_scale,
              Tensor accum,
              const Tensor &smooth_scale) {
    constexpr const char *op_name = "deepseek_v4_lmslim_rocblas_linear_w8a8_";
    check_tensors(output, input, weight_t, weight_scale, bias, q_input, input_scale, accum, smooth_scale, op_name);

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

        const auto &symbols = lmslim_rocblas_symbols();
        auto *stream = reinterpret_cast<hipStream_t>(context::getStream());
        auto handle = rocblas_handle_for_device(device_index);
        if (rocblas_set_stream(handle, stream) != rocblas_status_success) {
            throw std::runtime_error(std::string(op_name) + " failed to set rocBLAS stream.");
        }
        const bool transpose_weight = !weight_t->is_contiguous();
        symbols.gemm_rocblas_int8_gpu(stream,
                                      handle,
                                      reinterpret_cast<int *>(accum->data()),
                                      reinterpret_cast<const int8_t *>(q_input->data()),
                                      reinterpret_cast<const int8_t *>(weight_t->data()),
                                      static_cast<int>(input->size(0)),
                                      static_cast<int>(output->size(1)),
                                      static_cast<int>(input->size(1)),
                                      transpose_weight);
        deepseek_v4_lmslim_rocblas_linear_w8a8_impl::launch_apply_scales(output->data(),
                                                                         reinterpret_cast<const int32_t *>(accum->data()),
                                                                         reinterpret_cast<const float *>(input_scale->data()),
                                                                         reinterpret_cast<const float *>(weight_scale->data()),
                                                                         bias.has_value() ? bias.value()->data() : nullptr,
                                                                         static_cast<int64_t>(input->size(0)),
                                                                         static_cast<int64_t>(output->size(1)),
                                                                         output->dtype(),
                                                                         context::getStream());
    };

    if (profile_enabled()) {
        context::syncStream();
        const auto total_start = std::chrono::steady_clock::now();
        run_once();
        context::syncStream();
        const auto total_end = std::chrono::steady_clock::now();
        const auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();
        std::fprintf(stderr,
                     "[INFINICORE_DSV4_LMSLIM_ROCBLAS_LINEAR_PROFILE] M=%zu N=%zu K=%zu total_ms=%.6f\n",
                     input->size(0),
                     output->size(1),
                     input->size(1),
                     static_cast<double>(total_us) / 1000.0);
    } else {
        run_once();
    }
#else
    (void)output;
    (void)input;
    (void)weight_t;
    (void)weight_scale;
    (void)bias;
    (void)q_input;
    (void)input_scale;
    (void)accum;
    (void)smooth_scale;
    throw std::runtime_error("deepseek_v4_lmslim_rocblas_linear_w8a8_ requires a HYGON build.");
#endif
}

} // namespace

DeepseekV4LmslimRocblasLinearW8A8::DeepseekV4LmslimRocblasLinearW8A8(Tensor output,
                                                                     const Tensor &input,
                                                                     const Tensor &weight_t,
                                                                     const Tensor &weight_scale,
                                                                     std::optional<Tensor> bias,
                                                                     Tensor q_input,
                                                                     Tensor input_scale,
                                                                     Tensor accum,
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
                                 accum,
                                 smooth_scale);
}

void DeepseekV4LmslimRocblasLinearW8A8::execute(Tensor output,
                                                const Tensor &input,
                                                const Tensor &weight_t,
                                                const Tensor &weight_scale,
                                                std::optional<Tensor> bias,
                                                Tensor q_input,
                                                Tensor input_scale,
                                                Tensor accum,
                                                const Tensor &smooth_scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4LmslimRocblasLinearW8A8,
                                      output,
                                      input,
                                      weight_t,
                                      weight_scale,
                                      bias,
                                      q_input,
                                      input_scale,
                                      accum,
                                      smooth_scale);
}

namespace deepseek_v4_lmslim_rocblas_linear_w8a8_graph_impl {

struct PlannedMeta {
    graph::GraphTensor output;
    graph::GraphTensor input;
    graph::GraphTensor weight_t;
    graph::GraphTensor weight_scale;
    std::optional<graph::GraphTensor> bias;
    graph::GraphTensor q_input;
    graph::GraphTensor input_scale;
    graph::GraphTensor accum;
    graph::GraphTensor smooth_scale;
};

void *plan(Tensor output,
           const Tensor &input,
           const Tensor &weight_t,
           const Tensor &weight_scale,
           std::optional<Tensor> bias,
           Tensor q_input,
           Tensor input_scale,
           Tensor accum,
           const Tensor &smooth_scale) {
    check_tensors(output,
                  input,
                  weight_t,
                  weight_scale,
                  bias,
                  q_input,
                  input_scale,
                  accum,
                  smooth_scale,
                  "deepseek_v4_lmslim_rocblas_linear_w8a8_");
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
                           graph::GraphTensor(accum),
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
             planned->accum,
             planned->smooth_scale);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_lmslim_rocblas_linear_w8a8_graph_impl

namespace deepseek_v4_lmslim_rocblas_linear_w8a8_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4LmslimRocblasLinearW8A8,
                                       &deepseek_v4_lmslim_rocblas_linear_w8a8_graph_impl::plan,
                                       &deepseek_v4_lmslim_rocblas_linear_w8a8_graph_impl::run,
                                       &deepseek_v4_lmslim_rocblas_linear_w8a8_graph_impl::cleanup);
} // namespace deepseek_v4_lmslim_rocblas_linear_w8a8_register

void deepseek_v4_lmslim_rocblas_linear_w8a8_(Tensor output,
                                             const Tensor &input,
                                             const Tensor &weight_t,
                                             const Tensor &weight_scale,
                                             std::optional<Tensor> bias,
                                             Tensor q_input,
                                             Tensor input_scale,
                                             Tensor accum,
                                             const Tensor &smooth_scale) {
    DeepseekV4LmslimRocblasLinearW8A8::execute(output,
                                               input,
                                               weight_t,
                                               weight_scale,
                                               bias,
                                               q_input,
                                               input_scale,
                                               accum,
                                               smooth_scale);
}

} // namespace infinicore::op
