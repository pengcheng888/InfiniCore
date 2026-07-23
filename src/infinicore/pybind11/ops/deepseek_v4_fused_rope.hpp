#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_fused_rope.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void py_deepseek_v4_fused_rope_(Tensor query,
                                       py::object key,
                                       Tensor freqs_cis,
                                       Tensor positions,
                                       bool inverse) {
    std::optional<Tensor> key_tensor = std::nullopt;
    if (!key.is_none()) {
        key_tensor = key.cast<Tensor>();
    }
    op::deepseek_v4_fused_rope_(query, key_tensor, freqs_cis, positions, inverse);
}

inline void bind_deepseek_v4_fused_rope(py::module &m) {
    m.def("deepseek_v4_fused_rope_",
          &ops::py_deepseek_v4_fused_rope_,
          py::arg("query"),
          py::arg("key"),
          py::arg("freqs_cis"),
          py::arg("positions"),
          py::arg("inverse") = false,
          R"doc(DeepSeek-V4 pairwise fused RoPE matching SGLang fused_rope.)doc");
}

} // namespace infinicore::ops
