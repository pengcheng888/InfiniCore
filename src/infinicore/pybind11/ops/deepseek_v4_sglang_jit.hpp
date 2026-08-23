#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/device.hpp"
#include "infinicore/tensor.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <torch/csrc/utils/pybind.h>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#elif defined(ENABLE_NVIDIA_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

#include <optional>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace infinicore::ops {

namespace {

#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))

inline void check_deepseek_v4_jit_anchor(const Tensor &tensor) {
#if defined(ENABLE_HYGON_API)
    if (tensor->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_sglang_jit_call_ expects HYGON tensors in this build.");
    }
#else
    if (tensor->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_sglang_jit_call_ expects NVIDIA tensors in this build.");
    }
#endif
}

inline py::object to_py_torch_tensor(const Tensor &tensor) {
    return py::cast(infinicore::adaptor::to_aten_tensor(tensor));
}

inline py::object convert_sglang_jit_arg(py::handle arg) {
    if (arg.is_none()) {
        return py::none();
    }
    try {
        return to_py_torch_tensor(arg.cast<Tensor>());
    } catch (const py::cast_error &) {
        return py::reinterpret_borrow<py::object>(arg);
    }
}

class DeepseekV4JitStreamGuard {
public:
    explicit DeepseekV4JitStreamGuard(const Tensor &anchor) {
        check_deepseek_v4_jit_anchor(anchor);
#if defined(ENABLE_HYGON_API)
        hip_guard_.emplace(infinicore::adaptor::get_hip_stream());
#else
        cuda_guard_.emplace(infinicore::adaptor::get_cuda_stream());
#endif
    }

private:
#if defined(ENABLE_HYGON_API)
    std::optional<c10::hip::HIPStreamGuard> hip_guard_;
#else
    std::optional<c10::cuda::CUDAStreamGuard> cuda_guard_;
#endif
};

#endif

} // namespace

inline py::object py_deepseek_v4_sglang_jit_call_(const std::string &fn_name, Tensor anchor, py::args args) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    DeepseekV4JitStreamGuard guard(anchor);
    py::gil_scoped_acquire gil;

    py::tuple converted(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
        converted[i] = convert_sglang_jit_arg(args[i]);
    }
    py::module_ bridge = py::module_::import("infinicore.ops._deepseek_v4_sglang_jit");
    return bridge.attr(fn_name.c_str())(*converted);
#else
    (void)fn_name;
    (void)anchor;
    (void)args;
    throw std::runtime_error("deepseek_v4_sglang_jit_call_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

inline void bind_deepseek_v4_sglang_jit(py::module &m) {
    m.def("deepseek_v4_sglang_jit_call_",
          &ops::py_deepseek_v4_sglang_jit_call_,
          py::arg("fn_name"),
          py::arg("anchor"));
}

} // namespace infinicore::ops
