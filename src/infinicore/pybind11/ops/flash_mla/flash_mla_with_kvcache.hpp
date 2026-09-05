#pragma once

#include <utility>

#include <pybind11/pybind11.h>

#include "infinicore/ops/flash_mla/flash_mla_with_kvcache.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline std::optional<Tensor> py_optional_tensor_for_flash_mla_with_kvcache(py::object obj) {
    if (obj.is_none()) {
        return std::nullopt;
    }
    return obj.cast<Tensor>();
}

inline std::pair<Tensor, Tensor> py_flash_mla_with_kvcache(
    Tensor q,
    Tensor k_cache,
    py::object block_table,
    py::object cache_seqlens,
    int64_t head_dim_v,
    const op::flash_mla::FlashMLASchedMeta &tile_scheduler_metadata,
    py::object num_splits,
    py::object softmax_scale,
    bool causal,
    bool is_fp8_kvcache,
    py::object indices,
    py::object attn_sink,
    py::object extra_k_cache,
    py::object extra_indices_in_kvcache,
    py::object topk_length,
    py::object extra_topk_length) {
    return op::flash_mla::flash_mla_with_kvcache(
        q,
        k_cache,
        py_optional_tensor_for_flash_mla_with_kvcache(block_table),
        py_optional_tensor_for_flash_mla_with_kvcache(cache_seqlens),
        head_dim_v,
        tile_scheduler_metadata,
        py_optional_tensor_for_flash_mla_with_kvcache(num_splits),
        softmax_scale.is_none() ? std::nullopt : std::optional<double>(softmax_scale.cast<double>()),
        causal,
        is_fp8_kvcache,
        py_optional_tensor_for_flash_mla_with_kvcache(indices),
        py_optional_tensor_for_flash_mla_with_kvcache(attn_sink),
        py_optional_tensor_for_flash_mla_with_kvcache(extra_k_cache),
        py_optional_tensor_for_flash_mla_with_kvcache(extra_indices_in_kvcache),
        py_optional_tensor_for_flash_mla_with_kvcache(topk_length),
        py_optional_tensor_for_flash_mla_with_kvcache(extra_topk_length));
}

inline void bind_flash_mla_with_kvcache(py::module &m) {
    py::class_<op::flash_mla::FlashMLASchedMeta>(m, "FlashMLASchedMeta")
        .def(py::init<>())
        .def("has_sched_buffer", &op::flash_mla::FlashMLASchedMeta::has_sched_buffer)
        .def("has_valid_sched_meta", &op::flash_mla::FlashMLASchedMeta::has_valid_sched_meta)
        .def("has_sched_meta", &op::flash_mla::FlashMLASchedMeta::has_sched_meta);

    m.def("flash_mla_with_kvcache",
          &ops::py_flash_mla_with_kvcache,
          py::arg("q"),
          py::arg("k_cache"),
          py::arg("block_table"),
          py::arg("cache_seqlens"),
          py::arg("head_dim_v"),
          py::arg("tile_scheduler_metadata"),
          py::arg("num_splits") = py::none(),
          py::arg("softmax_scale") = py::none(),
          py::arg("causal") = false,
          py::arg("is_fp8_kvcache") = false,
          py::arg("indices") = py::none(),
          py::arg("attn_sink") = py::none(),
          py::arg("extra_k_cache") = py::none(),
          py::arg("extra_indices_in_kvcache") = py::none(),
          py::arg("topk_length") = py::none(),
          py::arg("extra_topk_length") = py::none(),
          R"doc(FlashMLA KV-cache forward. Returns out and lse.)doc");
}

} // namespace infinicore::ops
