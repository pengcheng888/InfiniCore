#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/embedding.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_embedding(py::module &m) {
    m.def("embedding",
          &op::embedding,
          py::arg("input"),
          py::arg("weight"),
          R"doc(Matrix multiplication of two tensors.)doc");
}

} // namespace infinicore::ops
