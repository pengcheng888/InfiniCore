#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_fast_topk.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline std::optional<Tensor> py_optional_tensor(py::object obj) {
    if (obj.is_none()) {
        return std::nullopt;
    }
    return obj.cast<Tensor>();
}

inline void py_deepseek_v4_fast_topk_(Tensor score,
                                      Tensor indices,
                                      Tensor lengths,
                                      py::object row_starts) {
    op::deepseek_v4_fast_topk_(score, indices, lengths, py_optional_tensor(row_starts));
}

inline void py_deepseek_v4_fast_topk_transform_fused_(Tensor score,
                                                      Tensor lengths,
                                                      Tensor dst_page_table,
                                                      Tensor src_page_table,
                                                      Tensor cu_seqlens_q,
                                                      py::object row_starts) {
    op::deepseek_v4_fast_topk_transform_fused_(
        score, lengths, dst_page_table, src_page_table, cu_seqlens_q, py_optional_tensor(row_starts));
}

inline void py_deepseek_v4_fast_topk_transform_ragged_fused_(Tensor score,
                                                             Tensor lengths,
                                                             Tensor topk_indices_ragged,
                                                             Tensor topk_indices_offset,
                                                             py::object row_starts) {
    op::deepseek_v4_fast_topk_transform_ragged_fused_(
        score, lengths, topk_indices_ragged, topk_indices_offset, py_optional_tensor(row_starts));
}

inline void bind_deepseek_v4_fast_topk(py::module &m) {
    m.def("deepseek_v4_fast_topk_",
          &ops::py_deepseek_v4_fast_topk_,
          py::arg("score"),
          py::arg("indices"),
          py::arg("lengths"),
          py::arg("row_starts") = py::none(),
          R"doc(DeepSeek-V4 SGLang fast topk bridge.)doc");
    m.def("deepseek_v4_fast_topk_transform_fused_",
          &ops::py_deepseek_v4_fast_topk_transform_fused_,
          py::arg("score"),
          py::arg("lengths"),
          py::arg("dst_page_table"),
          py::arg("src_page_table"),
          py::arg("cu_seqlens_q"),
          py::arg("row_starts") = py::none(),
          R"doc(DeepSeek-V4 SGLang fast topk + page-table transform bridge.)doc");
    m.def("deepseek_v4_fast_topk_transform_ragged_fused_",
          &ops::py_deepseek_v4_fast_topk_transform_ragged_fused_,
          py::arg("score"),
          py::arg("lengths"),
          py::arg("topk_indices_ragged"),
          py::arg("topk_indices_offset"),
          py::arg("row_starts") = py::none(),
          R"doc(DeepSeek-V4 SGLang fast topk + ragged transform bridge.)doc");
}

} // namespace infinicore::ops
