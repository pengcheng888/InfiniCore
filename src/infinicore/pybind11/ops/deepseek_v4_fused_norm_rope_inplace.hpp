#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_fused_norm_rope_inplace.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_fused_norm_rope_inplace(py::module &m) {
    m.def("deepseek_v4_fused_norm_rope_inplace_",
          &op::deepseek_v4_fused_norm_rope_inplace_,
          py::arg("input"),
          py::arg("norm_weight"),
          py::arg("epsilon"),
          py::arg("freqs_cis"),
          py::arg("positions"));
    m.def("deepseek_v4_fused_norm_rope_inplace",
          &op::deepseek_v4_fused_norm_rope_inplace,
          py::arg("input"),
          py::arg("norm_weight"),
          py::arg("epsilon"),
          py::arg("freqs_cis"),
          py::arg("positions"));
    m.def("deepseek_v4_fused_norm_rope_inplace_kernel_",
          &op::deepseek_v4_fused_norm_rope_inplace_kernel_,
          py::arg("input"),
          py::arg("norm_weight"),
          py::arg("epsilon"),
          py::arg("freqs_cis"),
          py::arg("positions"));
    m.def("deepseek_v4_fused_norm_rope_inplace_naive_",
          &op::deepseek_v4_fused_norm_rope_inplace_naive_,
          py::arg("input"),
          py::arg("norm_weight"),
          py::arg("epsilon"),
          py::arg("freqs_cis"),
          py::arg("positions"));
}

} // namespace infinicore::ops
