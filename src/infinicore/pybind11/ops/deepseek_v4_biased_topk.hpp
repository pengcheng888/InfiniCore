#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_biased_topk.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_biased_topk(py::module &m) {
    m.def("deepseek_v4_topk_naive_",
          &op::deepseek_v4_topk_naive_,
          py::arg("topk_weights"),
          py::arg("topk_indices"),
          py::arg("router_logits"),
          py::arg("correction_bias"),
          py::arg("renormalize"));
}

} // namespace infinicore::ops
