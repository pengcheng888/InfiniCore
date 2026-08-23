#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_indexer_compress_norm_rope_store.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_indexer_compress_norm_rope_store(py::module &m) {
    m.def("deepseek_v4_indexer_compress_norm_rope_store_",
          &op::deepseek_v4_indexer_compress_norm_rope_store_,
          py::arg("kv"),
          py::arg("norm_weight"),
          py::arg("epsilon"),
          py::arg("freqs_cis"),
          py::arg("positions"),
          py::arg("out_loc"),
          py::arg("kvcache"),
          py::arg("page_size"));
    m.def("deepseek_v4_indexer_compress_norm_rope_store_kernel_",
          &op::deepseek_v4_indexer_compress_norm_rope_store_kernel_,
          py::arg("kv"),
          py::arg("norm_weight"),
          py::arg("epsilon"),
          py::arg("freqs_cis"),
          py::arg("positions"),
          py::arg("out_loc"),
          py::arg("kvcache"),
          py::arg("page_size"));
}

} // namespace infinicore::ops
