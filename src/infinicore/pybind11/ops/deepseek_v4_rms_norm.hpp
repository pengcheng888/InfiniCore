#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_rms_norm.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_rms_norm(py::module &m) {
    m.def("deepseek_v4_rms_norm",
          &op::deepseek_v4_rms_norm,
          py::arg("input"),
          py::arg("weight"),
          py::arg("epsilon"),
          R"doc(DeepSeek-V4 RMSNorm wrapper.)doc");

    m.def("deepseek_v4_rms_norm_",
          &op::deepseek_v4_rms_norm_,
          py::arg("out"),
          py::arg("input"),
          py::arg("weight"),
          py::arg("epsilon"),
          R"doc(Out-variant DeepSeek-V4 RMSNorm wrapper.)doc");
}

} // namespace infinicore::ops
