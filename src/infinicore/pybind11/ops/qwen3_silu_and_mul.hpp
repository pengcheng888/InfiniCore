#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/qwen3_silu_and_mul.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_qwen3_silu_and_mul(py::module &m) {
    m.def("qwen3_silu_and_mul",
          &op::qwen3_silu_and_mul,
          py::arg("input"),
          R"doc(Qwen3 SiLU-and-mul activation wrapper.)doc");

    m.def("qwen3_silu_and_mul_",
          &op::qwen3_silu_and_mul_,
          py::arg("out"),
          py::arg("input"),
          R"doc(In-place Qwen3 SiLU-and-mul activation wrapper.)doc");
}

} // namespace infinicore::ops
