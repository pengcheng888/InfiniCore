#pragma once

#include "tensor.hpp"

#include <pybind11/pybind11.h>

namespace infinicore::py::tensor {
inline void bind(pybind11::module &m) {
    pybind11::class_<py::Tensor>(m, "Tensor")
        .def("copy_", &py::Tensor::copy_, pybind11::arg("src"))
        .def("to", &py::Tensor::to, pybind11::arg("device"));

    m.def("empty", &py::empty,
          pybind11::arg("shape"),
          pybind11::arg("dtype"),
          pybind11::arg("device"),
          pybind11::arg("pin_memory") = false);
    m.def("from_blob", &py::from_blob,
          pybind11::arg("raw_ptr"),
          pybind11::arg("shape"),
          pybind11::arg("dtype"),
          pybind11::arg("device"));
}
} // namespace infinicore::py::tensor
