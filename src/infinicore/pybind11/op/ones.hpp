#pragma once
#include "../tensor.hpp"
#include "infinicore.hpp"
#include <pybind11/pybind11.h>

#include "infinicore/op/ones.hpp"

namespace py = pybind11;

namespace infinicore::op {

Tensor ones_py(const Shape &shape, const DataType &dtype, const Device &device, bool pin_memory);

inline void bind_ones(py::module &m) {
    m.def("ones",
          &op::ones_py,
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("device"),
          pybind11::arg("pin_memory") = false,
          R"doc(Returns a tensor filled with the scalar value 1.)doc");
}

} // namespace infinicore::op
