#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/qwen3_rms_norm.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_qwen3_rms_norm(py::module &m) {
    m.def("qwen3_rms_norm",
          &op::qwen3_rms_norm,
          py::arg("x"),
          py::arg("weight"),
          py::arg("epsilon") = 1e-6f,
          R"doc(Qwen3 RMSNorm wrapper.)doc");

    m.def("qwen3_rms_norm_",
          &op::qwen3_rms_norm_,
          py::arg("y"),
          py::arg("x"),
          py::arg("weight"),
          py::arg("epsilon") = 1e-6f,
          R"doc(In-place Qwen3 RMSNorm wrapper.)doc");
}

} // namespace infinicore::ops
