#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/rope.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_rope(py::module &m) {
    m.def("rope",
          &op::rope,
          py::arg("x"),
          py::arg("pos_ids"),
          py::arg("sin_table"),
          py::arg("cos_table"),
          R"doc(TODO!.

Args:
    x: Input tensor
    weight: Scale weights
    epsilon: Small constant for numerical stability, default is 1e-5

Returns:
    Normalized tensor with same shape as input
)doc");

    m.def("rope_",
          &op::rope_,
          py::arg("y"),
          py::arg("x"),
          py::arg("pos_ids"),
          py::arg("sin_table"),
          py::arg("cos_table"),
          R"doc(TODO!.

Args:
    y: Output tensor
    x: Input tensor
    weight: Scale weights
    epsilon: Small constant for numerical stability, default is 1e-5
)doc");
}

} // namespace infinicore::ops
