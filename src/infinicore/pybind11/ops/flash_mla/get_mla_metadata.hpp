#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/flash_mla/get_mla_metadata.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline std::optional<int64_t> py_optional_i64_for_get_mla_metadata(py::object obj) {
    if (obj.is_none()) {
        return std::nullopt;
    }
    return obj.cast<int64_t>();
}

inline op::flash_mla::FlashMLASchedMeta py_get_mla_metadata(
    Tensor cache_seqlens,
    int64_t num_q_tokens_per_head_k,
    int64_t num_heads_k,
    py::object num_heads_q,
    bool is_fp8_kvcache,
    py::object topk,
    py::object sched_meta) {
    auto metadata = sched_meta.is_none()
                      ? op::flash_mla::FlashMLASchedMeta()
                      : sched_meta.cast<op::flash_mla::FlashMLASchedMeta>();
    op::flash_mla::get_mla_metadata_(
        metadata,
        cache_seqlens,
        num_q_tokens_per_head_k,
        num_heads_k,
        py_optional_i64_for_get_mla_metadata(num_heads_q),
        is_fp8_kvcache,
        py_optional_i64_for_get_mla_metadata(topk));
    return metadata;
}

inline void bind_flash_mla_get_mla_metadata(py::module &m) {
    m.def("get_mla_metadata",
          &ops::py_get_mla_metadata,
          py::arg("cache_seqlens"),
          py::arg("num_q_tokens_per_head_k"),
          py::arg("num_heads_k"),
          py::arg("num_heads_q") = py::none(),
          py::arg("is_fp8_kvcache") = false,
          py::arg("topk") = py::none(),
          py::arg("sched_meta") = py::none(),
          R"doc(Build or refresh FlashMLA scheduler metadata.)doc");
}

} // namespace infinicore::ops
