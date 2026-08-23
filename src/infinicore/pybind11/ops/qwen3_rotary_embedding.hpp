#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/qwen3_rotary_embedding.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void py_qwen3_rotary_embedding_(Tensor positions,
                                       Tensor query,
                                       py::object key,
                                       int head_size,
                                       Tensor cos_sin_cache,
                                       bool is_neox) {
    std::optional<Tensor> key_tensor = std::nullopt;
    if (!key.is_none()) {
        key_tensor = key.cast<Tensor>();
    }
    op::qwen3_rotary_embedding_(positions, query, key_tensor, head_size, cos_sin_cache, is_neox);
}

inline void bind_qwen3_rotary_embedding(py::module &m) {
    m.def("qwen3_rotary_embedding_",
          &ops::py_qwen3_rotary_embedding_,
          py::arg("positions"),
          py::arg("query"),
          py::arg("key"),
          py::arg("head_size"),
          py::arg("cos_sin_cache"),
          py::arg("is_neox") = true,
          R"doc(In-place Qwen3 rotary embedding backed by SGLang sgl_kernel.)doc");
}

} // namespace infinicore::ops

