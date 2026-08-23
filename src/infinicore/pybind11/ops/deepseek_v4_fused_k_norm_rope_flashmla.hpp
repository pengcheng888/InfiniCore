#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_fused_k_norm_rope_flashmla.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_fused_k_norm_rope_flashmla(py::module &m) {
    m.def("deepseek_v4_fused_k_norm_rope_flashmla_",
          &op::deepseek_v4_fused_k_norm_rope_flashmla_,
          py::arg("kv"),
          py::arg("kv_weight"),
          py::arg("epsilon"),
          py::arg("freqs_cis"),
          py::arg("positions"),
          py::arg("out_loc"),
          py::arg("kvcache"),
          py::arg("page_size"));
    m.def("deepseek_v4_fused_k_norm_rope_flashmla_kernel_",
          &op::deepseek_v4_fused_k_norm_rope_flashmla_kernel_,
          py::arg("kv"),
          py::arg("kv_weight"),
          py::arg("epsilon"),
          py::arg("freqs_cis"),
          py::arg("positions"),
          py::arg("out_loc"),
          py::arg("kvcache"),
          py::arg("page_size"));
}

} // namespace infinicore::ops
