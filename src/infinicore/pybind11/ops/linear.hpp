#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/linear.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_linear(py::module &m) {
    m.def("linear",
          &op::linear,
          py::arg("input"),
          py::arg("weight"),
          R"doc(Matrix multiplication of two tensors.)doc");

    m.def("linear_bias",
          &op::linear_bias,
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          R"doc(Matrix multiplication of two tensors.)doc");

    m.def("linear2",
          &op::linear2,
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          R"doc(Matrix multiplication of two tensors.)doc");
}

} // namespace infinicore::ops
