#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/qwen3_mha_varlen.hpp"

namespace py = pybind11;

namespace infinicore::ops {

Tensor py_qwen3_mha_varlen(Tensor q,
                           Tensor k,
                           Tensor v,
                           Tensor cum_seqlens_q,
                           Tensor cum_seqlens_k,
                           pybind11::object block_table,
                           int max_seqlen_q,
                           int max_seqlen_k,
                           pybind11::object alibi_slopes,
                           float scale) {
    std::optional<Tensor> block_table_tensor = std::nullopt;
    if (!block_table.is_none()) {
        block_table_tensor = block_table.cast<Tensor>();
    }
    std::optional<Tensor> alibi_slopes_tensor = std::nullopt;
    if (!alibi_slopes.is_none()) {
        alibi_slopes_tensor = alibi_slopes.cast<Tensor>();
    }
    return op::qwen3_mha_varlen(q, k, v, cum_seqlens_q, cum_seqlens_k, block_table_tensor, max_seqlen_q, max_seqlen_k, alibi_slopes_tensor, scale);
}

void py_qwen3_mha_varlen_(Tensor out,
                          Tensor q,
                          Tensor k,
                          Tensor v,
                          Tensor cum_seqlens_q,
                          Tensor cum_seqlens_k,
                          pybind11::object block_table,
                          int max_seqlen_q,
                          int max_seqlen_k,
                          pybind11::object alibi_slopes,
                          float scale) {
    std::optional<Tensor> block_table_tensor = std::nullopt;
    if (!block_table.is_none()) {
        block_table_tensor = block_table.cast<Tensor>();
    }
    std::optional<Tensor> alibi_slopes_tensor = std::nullopt;
    if (!alibi_slopes.is_none()) {
        alibi_slopes_tensor = alibi_slopes.cast<Tensor>();
    }
    op::qwen3_mha_varlen_(out, q, k, v, cum_seqlens_q, cum_seqlens_k, block_table_tensor, max_seqlen_q, max_seqlen_k, alibi_slopes_tensor, scale);
}

inline void bind_qwen3_mha_varlen(py::module &m) {
    m.def("qwen3_mha_varlen", &ops::py_qwen3_mha_varlen,
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("cum_seqlens_q"), py::arg("cum_seqlens_k"),
          py::arg("block_table"), py::arg("max_seqlen_q"), py::arg("max_seqlen_k"), py::arg("alibi_slopes"), py::arg("scale"),
          R"doc(Qwen3 variable-length attention wrapper backed by InfiniCore FlashAttention integration.)doc");
    m.def("qwen3_mha_varlen_", &ops::py_qwen3_mha_varlen_,
          py::arg("out"), py::arg("q"), py::arg("k"), py::arg("v"), py::arg("cum_seqlens_q"), py::arg("cum_seqlens_k"),
          py::arg("block_table"), py::arg("max_seqlen_q"), py::arg("max_seqlen_k"), py::arg("alibi_slopes"), py::arg("scale"),
          R"doc(In-place Qwen3 variable-length attention wrapper.)doc");
}

} // namespace infinicore::ops
