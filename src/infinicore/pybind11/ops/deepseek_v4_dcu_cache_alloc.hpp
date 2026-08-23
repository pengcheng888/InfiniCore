#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_dcu_cache_alloc.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_dcu_cache_alloc(py::module &m) {
    m.def("deepseek_v4_dcu_alloc_decode_kernel_",
          &op::deepseek_v4_dcu_alloc_decode_kernel_,
          py::arg("seq_lens"),
          py::arg("last_loc"),
          py::arg("free_page"),
          py::arg("out_indices"),
          py::arg("bs"),
          py::arg("page_size"),
          R"doc(DeepSeek-V4 SGLang DCU decode cache allocation bridge.)doc");
    m.def("deepseek_v4_dcu_alloc_extend_kernel_",
          &op::deepseek_v4_dcu_alloc_extend_kernel_,
          py::arg("pre_lens"),
          py::arg("seq_lens"),
          py::arg("last_loc"),
          py::arg("free_page"),
          py::arg("out_indices"),
          py::arg("bs"),
          py::arg("page_size"),
          R"doc(DeepSeek-V4 SGLang DCU extend cache allocation bridge.)doc");
}

} // namespace infinicore::ops
