#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/qwen3_mha_kvcache.hpp"

namespace py = pybind11;

namespace infinicore::ops {

Tensor py_qwen3_mha_kvcache(Tensor q,
                            Tensor k_cache,
                            Tensor v_cache,
                            Tensor seqlens_k,
                            Tensor block_table,
                            pybind11::object alibi_slopes,
                            float scale) {
    std::optional<Tensor> alibi_slopes_tensor = std::nullopt;
    if (!alibi_slopes.is_none()) {
        alibi_slopes_tensor = alibi_slopes.cast<Tensor>();
    }
    return op::qwen3_mha_kvcache(q, k_cache, v_cache, seqlens_k, block_table, alibi_slopes_tensor, scale);
}

void py_qwen3_mha_kvcache_(Tensor out,
                           Tensor q,
                           Tensor k_cache,
                           Tensor v_cache,
                           Tensor seqlens_k,
                           Tensor block_table,
                           pybind11::object alibi_slopes,
                           float scale) {
    std::optional<Tensor> alibi_slopes_tensor = std::nullopt;
    if (!alibi_slopes.is_none()) {
        alibi_slopes_tensor = alibi_slopes.cast<Tensor>();
    }
    op::qwen3_mha_kvcache_(out, q, k_cache, v_cache, seqlens_k, block_table, alibi_slopes_tensor, scale);
}

inline void bind_qwen3_mha_kvcache(py::module &m) {
    m.def("qwen3_mha_kvcache", &ops::py_qwen3_mha_kvcache,
          py::arg("q"), py::arg("k_cache"), py::arg("v_cache"), py::arg("seqlens_k"), py::arg("block_table"), py::arg("alibi_slopes"), py::arg("scale"),
          R"doc(Qwen3 KV-cache attention wrapper backed by InfiniCore FlashAttention integration.)doc");
    m.def("qwen3_mha_kvcache_", &ops::py_qwen3_mha_kvcache_,
          py::arg("out"), py::arg("q"), py::arg("k_cache"), py::arg("v_cache"), py::arg("seqlens_k"), py::arg("block_table"), py::arg("alibi_slopes"), py::arg("scale"),
          R"doc(In-place Qwen3 KV-cache attention wrapper.)doc");
}

} // namespace infinicore::ops
