#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_topk_transform_512.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_topk_transform_512(py::module &m) {
    m.def("deepseek_v4_topk_transform_512_kernel_",
          &op::deepseek_v4_topk_transform_512_kernel_,
          py::arg("scores"),
          py::arg("seq_lens"),
          py::arg("page_table"),
          py::arg("out_page_indices"),
          py::arg("page_size") = 64);
    m.def("deepseek_v4_topk_transform_512_sglang_kernel_",
          &op::deepseek_v4_topk_transform_512_sglang_kernel_,
          py::arg("scores"),
          py::arg("seq_lens"),
          py::arg("page_table"),
          py::arg("out_page_indices"),
          py::arg("page_size") = 64);
}

} // namespace infinicore::ops
