#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_rmsnorm_self.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_rmsnorm_self(py::module &m) {
    m.def("deepseek_v4_rmsnorm_self",
          &op::deepseek_v4_rmsnorm_self,
          py::arg("input"),
          py::arg("epsilon"),
          R"doc(DeepSeek-V4 parameterless RMSNorm over the last dimension.)doc");

    m.def("deepseek_v4_rmsnorm_self_",
          &op::deepseek_v4_rmsnorm_self_,
          py::arg("out"),
          py::arg("input"),
          py::arg("epsilon"),
          R"doc(Out-variant DeepSeek-V4 parameterless RMSNorm over the last dimension.)doc");
}

} // namespace infinicore::ops
