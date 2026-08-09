#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang(py::module &m) {
    m.def("deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang_",
          &op::deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang_,
          py::arg("q_input"),
          py::arg("q_fp8"),
          py::arg("weight"),
          py::arg("weights_out"),
          py::arg("weight_scale"),
          py::arg("freqs_cis"),
          py::arg("positions"));
    m.def("deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang_kernel_",
          &op::deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang_kernel_,
          py::arg("q_input"),
          py::arg("q_fp8"),
          py::arg("weight"),
          py::arg("weights_out"),
          py::arg("weight_scale"),
          py::arg("freqs_cis"),
          py::arg("positions"));
}

} // namespace infinicore::ops
