#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_concat_and_cache_mla.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_concat_and_cache_mla(py::module &m) {
    m.def("deepseek_v4_concat_and_cache_mla_",
          &op::deepseek_v4_concat_and_cache_mla_,
          py::arg("kv_c"),
          py::arg("k_pe"),
          py::arg("kv_cache"),
          py::arg("slot_mapping"),
          py::arg("kv_cache_dtype"),
          py::arg("scale"),
          R"doc(DeepSeek-V4 VLLM MLA concat-and-cache bridge.)doc");
}

} // namespace infinicore::ops
