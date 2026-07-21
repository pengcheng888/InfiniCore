#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_add_rms_norm.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_add_rms_norm(py::module &m) {
    m.def("deepseek_v4_add_rms_norm",
          &op::deepseek_v4_add_rms_norm,
          py::arg("a"),
          py::arg("b"),
          py::arg("weight"),
          py::arg("epsilon"),
          R"doc(DeepSeek-V4 fused add + RMSNorm wrapper.)doc");

    m.def("deepseek_v4_add_rms_norm_",
          &op::deepseek_v4_add_rms_norm_,
          py::arg("out"),
          py::arg("residual"),
          py::arg("a"),
          py::arg("b"),
          py::arg("weight"),
          py::arg("epsilon"),
          R"doc(Out-variant DeepSeek-V4 fused add + RMSNorm wrapper.)doc");

    m.def("deepseek_v4_add_rms_norm_inplace",
          &op::deepseek_v4_add_rms_norm_inplace,
          py::arg("input"),
          py::arg("residual"),
          py::arg("weight"),
          py::arg("epsilon"),
          R"doc(In-place DeepSeek-V4 fused add + RMSNorm wrapper.)doc");
}

} // namespace infinicore::ops
