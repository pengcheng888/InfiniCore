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
          R"doc(Default parameterless RMSNorm for DeepSeek-V4 attention.)doc");

    m.def("deepseek_v4_rmsnorm_self_",
          &op::deepseek_v4_rmsnorm_self_,
          py::arg("out"),
          py::arg("input"),
          py::arg("epsilon"),
          R"doc(Out-variant default parameterless RMSNorm for DeepSeek-V4 attention.)doc");

    m.def("deepseek_v4_rmsnorm_self_naive",
          &op::deepseek_v4_rmsnorm_self_naive,
          py::arg("input"),
          py::arg("epsilon"));
    m.def("deepseek_v4_rmsnorm_self_naive_",
          &op::deepseek_v4_rmsnorm_self_naive_,
          py::arg("out"),
          py::arg("input"),
          py::arg("epsilon"));
    m.def("deepseek_v4_rmsnorm_self_kernel",
          &op::deepseek_v4_rmsnorm_self_kernel,
          py::arg("input"),
          py::arg("epsilon"));
    m.def("deepseek_v4_rmsnorm_self_kernel_",
          &op::deepseek_v4_rmsnorm_self_kernel_,
          py::arg("out"),
          py::arg("input"),
          py::arg("epsilon"));
}

} // namespace infinicore::ops
