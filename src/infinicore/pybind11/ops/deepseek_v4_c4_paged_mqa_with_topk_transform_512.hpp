#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_c4_paged_mqa_with_topk_transform_512.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_c4_paged_mqa_with_topk_transform_512(py::module &m) {
    m.def("deepseek_v4_c4_paged_mqa_with_topk_transform_512_",
          &op::deepseek_v4_c4_paged_mqa_with_topk_transform_512_,
          py::arg("q_fp8"),
          py::arg("fused_weights"),
          py::arg("indexer_kv_cache_raw"),
          py::arg("c4_seq_lens"),
          py::arg("page_table"),
          py::arg("out_page_indices"),
          py::arg("max_c4_seq_len"),
          py::arg("page_size") = 64,
          py::arg("clean_logits") = false);
}

} // namespace infinicore::ops
