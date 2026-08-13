#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_compress_fused_norm_rope.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_compress_fused_norm_rope(py::module &m) {
    m.def("deepseek_v4_compress_fused_norm_rope_",
          &op::deepseek_v4_compress_fused_norm_rope_,
          py::arg("input"),
          py::arg("norm_weight"),
          py::arg("epsilon"),
          py::arg("freqs_cis"),
          py::arg("positions"),
          R"doc(Default fused norm plus last-64 RoPE for DeepSeek-V4 compressed attention.)doc");
    m.def("deepseek_v4_compress_fused_norm_rope_naive_",
          &op::deepseek_v4_compress_fused_norm_rope_naive_,
          py::arg("input"),
          py::arg("norm_weight"),
          py::arg("epsilon"),
          py::arg("freqs_cis"),
          py::arg("positions"));
    m.def("deepseek_v4_compress_fused_norm_rope_kernel_",
          &op::deepseek_v4_compress_fused_norm_rope_kernel_,
          py::arg("input"),
          py::arg("norm_weight"),
          py::arg("epsilon"),
          py::arg("freqs_cis"),
          py::arg("positions"));
}

} // namespace infinicore::ops
