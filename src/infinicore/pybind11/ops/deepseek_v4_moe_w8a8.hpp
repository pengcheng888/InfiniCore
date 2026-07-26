#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_moe_w8a8.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_moe_w8a8(py::module &m) {
    m.def("deepseek_v4_moe_w8a8_",
          &op::deepseek_v4_moe_w8a8_,
          py::arg("y"),
          py::arg("x"),
          py::arg("topk_weights"),
          py::arg("topk_indices"),
          py::arg("w13"),
          py::arg("w13_scale"),
          py::arg("w2"),
          py::arg("w2_scale"),
          py::arg("swiglu_limit"));
}

} // namespace infinicore::ops
