#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_silu_and_mul.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_silu_and_mul(py::module &m) {
    m.def("deepseek_v4_silu_and_mul",
          &op::deepseek_v4_silu_and_mul,
          py::arg("input"),
          R"doc(Default DeepSeek-V4 SiLU-and-mul activation.)doc");

    m.def("deepseek_v4_silu_and_mul_",
          &op::deepseek_v4_silu_and_mul_,
          py::arg("out"),
          py::arg("input"),
          R"doc(Out-variant default DeepSeek-V4 SiLU-and-mul activation.)doc");

    m.def("deepseek_v4_silu_and_mul_dispatcher_",
          &op::deepseek_v4_silu_and_mul_dispatcher_,
          py::arg("out"),
          py::arg("input"),
          R"doc(SGLang dispatcher DeepSeek-V4 SiLU-and-mul activation.)doc");
}

} // namespace infinicore::ops
