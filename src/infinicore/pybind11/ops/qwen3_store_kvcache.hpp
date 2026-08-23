#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/qwen3_store_kvcache.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_qwen3_store_kvcache(py::module &m) {
    m.def("qwen3_store_kvcache_",
          &op::qwen3_store_kvcache_,
          py::arg("k"),
          py::arg("v"),
          py::arg("k_cache"),
          py::arg("v_cache"),
          py::arg("indices"),
          R"doc(Store Qwen3 K/V rows into paged KV cache on the current InfiniCore stream.)doc");
}

} // namespace infinicore::ops

