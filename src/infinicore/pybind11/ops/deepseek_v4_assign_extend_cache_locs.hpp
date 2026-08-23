#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_assign_extend_cache_locs.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_assign_extend_cache_locs(py::module &m) {
    m.def("deepseek_v4_assign_extend_cache_locs_",
          &op::deepseek_v4_assign_extend_cache_locs_,
          py::arg("req_pool_indices"),
          py::arg("req_to_token"),
          py::arg("start_offset"),
          py::arg("end_offset"),
          py::arg("out_cache_loc"),
          py::arg("pool_len"),
          py::arg("bs"),
          R"doc(DeepSeek-V4 SGLang extend-cache location assignment.)doc");
}

} // namespace infinicore::ops
