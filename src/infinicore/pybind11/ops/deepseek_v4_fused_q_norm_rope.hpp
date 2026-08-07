#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_fused_q_norm_rope.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_fused_q_norm_rope(py::module &m) {
    m.def("deepseek_v4_fused_q_norm_rope_",
          &op::deepseek_v4_fused_q_norm_rope_,
          py::arg("q_out"),
          py::arg("q_input"),
          py::arg("epsilon"),
          py::arg("freqs_cis"),
          py::arg("positions"));
    m.def("deepseek_v4_fused_q_norm_rope_kernel_",
          &op::deepseek_v4_fused_q_norm_rope_kernel_,
          py::arg("q_out"),
          py::arg("q_input"),
          py::arg("epsilon"),
          py::arg("freqs_cis"),
          py::arg("positions"));
    m.def("deepseek_v4_fused_q_norm_rope_naive_",
          &op::deepseek_v4_fused_q_norm_rope_naive_,
          py::arg("q_out"),
          py::arg("q_input"),
          py::arg("epsilon"),
          py::arg("freqs_cis"),
          py::arg("positions"));
}

} // namespace infinicore::ops
