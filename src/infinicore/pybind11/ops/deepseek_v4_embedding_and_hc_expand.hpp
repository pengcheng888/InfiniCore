#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_embedding_and_hc_expand.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_embedding_and_hc_expand(py::module &m) {
    m.def("deepseek_v4_embedding_and_hc_expand",
          &op::deepseek_v4_embedding_and_hc_expand,
          py::arg("input"),
          py::arg("weight"),
          py::arg("hc_mult"),
          R"doc(Default DeepSeek-V4 embedding lookup plus contiguous HC expansion.)doc");

    m.def("deepseek_v4_embedding_and_hc_expand_",
          &op::deepseek_v4_embedding_and_hc_expand_,
          py::arg("out"),
          py::arg("input"),
          py::arg("weight"),
          py::arg("hc_mult"),
          R"doc(Out-variant default DeepSeek-V4 embedding lookup plus contiguous HC expansion.)doc");

    m.def("deepseek_v4_embedding_and_hc_expand_kernel",
          &op::deepseek_v4_embedding_and_hc_expand_kernel,
          py::arg("input"),
          py::arg("weight"),
          py::arg("hc_mult"),
          R"doc(Native kernel DeepSeek-V4 embedding lookup plus contiguous HC expansion.)doc");

    m.def("deepseek_v4_embedding_and_hc_expand_kernel_",
          &op::deepseek_v4_embedding_and_hc_expand_kernel_,
          py::arg("out"),
          py::arg("input"),
          py::arg("weight"),
          py::arg("hc_mult"),
          R"doc(Out-variant native kernel DeepSeek-V4 embedding lookup plus contiguous HC expansion.)doc");

    m.def("deepseek_v4_embedding_and_hc_expand_aten",
          &op::deepseek_v4_embedding_and_hc_expand_aten,
          py::arg("input"),
          py::arg("weight"),
          py::arg("hc_mult"),
          R"doc(ATen reference DeepSeek-V4 embedding lookup plus contiguous HC expansion.)doc");

    m.def("deepseek_v4_embedding_and_hc_expand_aten_",
          &op::deepseek_v4_embedding_and_hc_expand_aten_,
          py::arg("out"),
          py::arg("input"),
          py::arg("weight"),
          py::arg("hc_mult"),
          R"doc(Out-variant ATen reference DeepSeek-V4 embedding lookup plus contiguous HC expansion.)doc");

}

} // namespace infinicore::ops
