#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_linear_bf16_fp32.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_linear_bf16_fp32(py::module &m) {
    m.def("deepseek_v4_linear_bf16_fp32",
          &op::deepseek_v4_linear_bf16_fp32,
          py::arg("input"),
          py::arg("weight"),
          R"doc(Default DeepSeek-V4 BF16xBF16 -> FP32 linear path.)doc");

    m.def("deepseek_v4_linear_bf16_fp32_",
          &op::deepseek_v4_linear_bf16_fp32_,
          py::arg("out"),
          py::arg("input"),
          py::arg("weight"),
          R"doc(Out-variant default DeepSeek-V4 BF16xBF16 -> FP32 linear path.)doc");

    m.def("deepseek_v4_linear_bf16_fp32_aten",
          &op::deepseek_v4_linear_bf16_fp32_aten,
          py::arg("input"),
          py::arg("weight"));
    m.def("deepseek_v4_linear_bf16_fp32_aten_",
          &op::deepseek_v4_linear_bf16_fp32_aten_,
          py::arg("out"),
          py::arg("input"),
          py::arg("weight"));
    m.def("deepseek_v4_linear_bf16_fp32_kernel",
          &op::deepseek_v4_linear_bf16_fp32_kernel,
          py::arg("input"),
          py::arg("weight"));
    m.def("deepseek_v4_linear_bf16_fp32_kernel_",
          &op::deepseek_v4_linear_bf16_fp32_kernel_,
          py::arg("out"),
          py::arg("input"),
          py::arg("weight"));
}

} // namespace infinicore::ops
