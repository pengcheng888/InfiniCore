#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_fused_qk_norm_rope.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_fused_qk_norm_rope(py::module &m) {
    m.def("deepseek_v4_fused_qk_norm_rope_",
          &op::deepseek_v4_fused_qk_norm_rope_,
          py::arg("qkv"),
          py::arg("num_heads_q"),
          py::arg("num_heads_k"),
          py::arg("num_heads_v"),
          py::arg("head_dim"),
          py::arg("eps"),
          py::arg("q_weight"),
          py::arg("k_weight"),
          py::arg("cos_sin_cache"),
          py::arg("is_neox"),
          py::arg("position_ids"),
          R"doc(DeepSeek-V4 VLLM fused Q/K RMSNorm + RoPE bridge.)doc");
}

} // namespace infinicore::ops
