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
          R"doc(DeepSeek-V4 BF16xBF16 -> FP32 linear kernel path, weight layout [out_features, in_features].)doc");

    m.def("deepseek_v4_linear_bf16_fp32_",
          &op::deepseek_v4_linear_bf16_fp32_,
          py::arg("out"),
          py::arg("input"),
          py::arg("weight"),
          R"doc(Out-variant DeepSeek-V4 BF16xBF16 -> FP32 linear kernel path.)doc");

    m.def("deepseek_v4_linear_bf16_fp32_naive",
          &op::deepseek_v4_linear_bf16_fp32_naive,
          py::arg("input"),
          py::arg("weight"),
          R"doc(DeepSeek-V4 BF16xBF16 -> FP32 linear ATen reference path.)doc");

    m.def("deepseek_v4_linear_bf16_fp32_naive_",
          &op::deepseek_v4_linear_bf16_fp32_naive_,
          py::arg("out"),
          py::arg("input"),
          py::arg("weight"),
          R"doc(Out-variant DeepSeek-V4 BF16xBF16 -> FP32 linear ATen reference path.)doc");

    m.def("deepseek_v4_linear_bf16_fp32_kernel",
          &op::deepseek_v4_linear_bf16_fp32_kernel,
          py::arg("input"),
          py::arg("weight"),
          R"doc(DeepSeek-V4 BF16xBF16 -> FP32 linear Hygon/NVIDIA kernel path.)doc");

    m.def("deepseek_v4_linear_bf16_fp32_kernel_",
          &op::deepseek_v4_linear_bf16_fp32_kernel_,
          py::arg("out"),
          py::arg("input"),
          py::arg("weight"),
          R"doc(Out-variant DeepSeek-V4 BF16xBF16 -> FP32 linear Hygon/NVIDIA kernel path.)doc");
}

} // namespace infinicore::ops
