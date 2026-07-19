#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/qwen3_fused_qk_norm_rope.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_qwen3_fused_qk_norm_rope(py::module &m) {
    m.def("qwen3_fused_qk_norm_rope_",
          &op::qwen3_fused_qk_norm_rope_,
          py::arg("qkv"),
          py::arg("num_heads_q"),
          py::arg("num_heads_k"),
          py::arg("num_heads_v"),
          py::arg("head_dim"),
          py::arg("eps"),
          py::arg("q_weight"),
          py::arg("k_weight"),
          py::arg("base"),
          py::arg("is_neox"),
          py::arg("position_ids"),
          py::arg("factor") = 1.0f,
          py::arg("low") = 0.0f,
          py::arg("high") = 0.0f,
          py::arg("attention_factor") = 1.0f,
          py::arg("rotary_dim") = 0,
          R"doc(In-place Qwen3 fused Q/K RMSNorm + RoPE backed by SGLang sgl_kernel.)doc");
}

} // namespace infinicore::ops

