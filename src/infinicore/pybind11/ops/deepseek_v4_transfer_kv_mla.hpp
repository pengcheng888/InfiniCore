#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_transfer_kv_mla.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_transfer_kv_mla(py::module &m) {
    m.def("deepseek_v4_transfer_kv_per_layer_mla_",
          &op::deepseek_v4_transfer_kv_per_layer_mla_,
          py::arg("src"),
          py::arg("dst"),
          py::arg("src_indices"),
          py::arg("dst_indices"),
          py::arg("item_size"),
          py::arg("block_quota") = 2,
          py::arg("num_warps_per_block") = 16,
          R"doc(DeepSeek-V4 SGLang per-layer MLA KV transfer bridge.)doc");
    m.def("deepseek_v4_transfer_kv_per_layer_mla_pf_lf_",
          &op::deepseek_v4_transfer_kv_per_layer_mla_pf_lf_,
          py::arg("src"),
          py::arg("dst"),
          py::arg("src_indices"),
          py::arg("dst_indices"),
          py::arg("layer_id"),
          py::arg("item_size"),
          py::arg("src_layout_dim"),
          py::arg("block_quota") = 2,
          py::arg("num_warps_per_block") = 16,
          R"doc(DeepSeek-V4 SGLang page-first to layer-first per-layer MLA KV transfer bridge.)doc");
}

} // namespace infinicore::ops
