#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/qwen3_add_rms_norm.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_qwen3_add_rms_norm(py::module &m) {
    m.def("qwen3_add_rms_norm",
          &op::qwen3_add_rms_norm,
          py::arg("a"),
          py::arg("b"),
          py::arg("weight"),
          py::arg("epsilon") = 1e-6f,
          R"doc(Qwen3 fused Add + RMSNorm wrapper.)doc");

    m.def("qwen3_add_rms_norm_",
          &op::qwen3_add_rms_norm_,
          py::arg("out"),
          py::arg("residual"),
          py::arg("a"),
          py::arg("b"),
          py::arg("weight"),
          py::arg("epsilon") = 1e-6f,
          R"doc(In-place Qwen3 fused Add + RMSNorm wrapper.)doc");

    m.def("qwen3_add_rms_norm_inplace",
          &op::qwen3_add_rms_norm_inplace,
          py::arg("input"),
          py::arg("residual"),
          py::arg("weight"),
          py::arg("epsilon") = 1e-6f,
          R"doc(Qwen3 fused Add + RMSNorm wrapper that updates input and residual.)doc");
}

} // namespace infinicore::ops
