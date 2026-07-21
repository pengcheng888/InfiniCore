#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_create_chunked_prefix_cache_kv_indices.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_create_chunked_prefix_cache_kv_indices(py::module &m) {
    m.def("deepseek_v4_create_chunked_prefix_cache_kv_indices_",
          &op::deepseek_v4_create_chunked_prefix_cache_kv_indices_,
          py::arg("req_to_token"),
          py::arg("req_pool_indices"),
          py::arg("chunk_starts"),
          py::arg("chunk_seq_lens"),
          py::arg("chunk_cu_seq_lens"),
          py::arg("chunk_kv_indices"),
          py::arg("col_num"),
          py::arg("bs"),
          R"doc(DeepSeek-V4 SGLang chunked-prefix KV index creation.)doc");
}

} // namespace infinicore::ops
