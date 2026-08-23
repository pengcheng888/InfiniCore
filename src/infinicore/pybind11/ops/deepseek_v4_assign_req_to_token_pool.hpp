#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_assign_req_to_token_pool.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_assign_req_to_token_pool(py::module &m) {
    m.def("deepseek_v4_assign_req_to_token_pool_",
          &op::deepseek_v4_assign_req_to_token_pool_,
          py::arg("req_pool_indices"),
          py::arg("req_to_token"),
          py::arg("allocate_lens"),
          py::arg("new_allocate_lens"),
          py::arg("out_cache_loc"),
          py::arg("shape"),
          py::arg("bs"),
          R"doc(DeepSeek-V4 SGLang req-to-token pool assignment.)doc");
}

} // namespace infinicore::ops
