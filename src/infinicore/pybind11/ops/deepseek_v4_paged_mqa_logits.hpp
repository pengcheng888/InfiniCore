#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_paged_mqa_logits.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_paged_mqa_logits(py::module &m) {
    m.def("deepseek_v4_paged_mqa_logits_metadata_",
          &op::deepseek_v4_paged_mqa_logits_metadata_,
          py::arg("context_lens"),
          py::arg("schedule_meta"),
          py::arg("block_kv"),
          py::arg("num_sms"));

    m.def("deepseek_v4_paged_mqa_logits_",
          &op::deepseek_v4_paged_mqa_logits_,
          py::arg("q"),
          py::arg("fused_kv_cache"),
          py::arg("weights"),
          py::arg("context_lens"),
          py::arg("block_table"),
          py::arg("schedule_meta"),
          py::arg("logits"),
          py::arg("max_context_len"),
          py::arg("clean_logits") = true);
}

} // namespace infinicore::ops
