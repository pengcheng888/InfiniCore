#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_fused_q_indexer_rope_hadamard_quant.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_fused_q_indexer_rope_hadamard_quant(py::module &m) {
    m.def("deepseek_v4_fused_q_indexer_rope_hadamard_quant_",
          &op::deepseek_v4_fused_q_indexer_rope_hadamard_quant_,
          py::arg("q"),
          py::arg("indexer_weights"),
          py::arg("q_fp8"),
          py::arg("q_scale"),
          py::arg("fused_weights"),
          py::arg("weight_scale"),
          py::arg("freqs_cis"),
          py::arg("positions"));
    m.def("deepseek_v4_fused_q_indexer_rope_hadamard_quant_kernel_",
          &op::deepseek_v4_fused_q_indexer_rope_hadamard_quant_kernel_,
          py::arg("q"),
          py::arg("indexer_weights"),
          py::arg("q_fp8"),
          py::arg("q_scale"),
          py::arg("fused_weights"),
          py::arg("weight_scale"),
          py::arg("freqs_cis"),
          py::arg("positions"));
}

} // namespace infinicore::ops
