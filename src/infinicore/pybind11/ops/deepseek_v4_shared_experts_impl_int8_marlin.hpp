#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_shared_experts_impl_int8_marlin.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_shared_experts_impl_int8_marlin(py::module &m) {
    m.def("deepseek_v4_shared_experts_impl_int8_marlin_",
          &op::deepseek_v4_shared_experts_impl_int8_marlin_,
          py::arg("output"),
          py::arg("hidden_states"),
          py::arg("w1"),
          py::arg("w2"),
          py::arg("w1_scale"),
          py::arg("w2_scale"),
          py::arg("gemm1_mode") = -1,
          py::arg("gemm2_mode") = -1,
          py::arg("delta") = 1,
          R"doc(DeepSeek-V4 INT8 Marlin shared MLP implementation for a single shared expert.)doc");
}

} // namespace infinicore::ops
