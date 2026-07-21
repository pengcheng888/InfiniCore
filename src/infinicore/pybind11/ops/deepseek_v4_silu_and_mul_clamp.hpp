#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_silu_and_mul_clamp.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_silu_and_mul_clamp(py::module &m) {
    m.def("deepseek_v4_silu_and_mul_clamp",
          &op::deepseek_v4_silu_and_mul_clamp,
          py::arg("input"),
          py::arg("swiglu_limit"),
          R"doc(DeepSeek-V4 SiLU-and-mul with SwiGLU clamp.)doc");

    m.def("deepseek_v4_silu_and_mul_clamp_",
          &op::deepseek_v4_silu_and_mul_clamp_,
          py::arg("out"),
          py::arg("input"),
          py::arg("swiglu_limit"),
          R"doc(Out-variant DeepSeek-V4 SiLU-and-mul with SwiGLU clamp.)doc");
}

} // namespace infinicore::ops
