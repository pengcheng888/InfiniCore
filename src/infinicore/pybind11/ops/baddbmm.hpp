#pragma once

#include "infinicore/ops/baddbmm.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

Tensor py_baddbmm(pybind11::object input, Tensor mat1, Tensor mat2, float beta, float alpha) {

    std::optional<Tensor> input_tensor = std::nullopt;
    if (!input.is_none()) {
        input_tensor = input.cast<Tensor>();
    }
    return op::baddbmm(input_tensor, mat1, mat2, beta, alpha);
}

void py_baddbmm_(Tensor out, pybind11::object input, Tensor mat1, Tensor mat2, float beta, float alpha) {

    std::optional<Tensor> input_tensor = std::nullopt;
    if (!input.is_none()) {
        input_tensor = input.cast<Tensor>();
    }

    op::baddbmm_(out, input_tensor, mat1, mat2, beta, alpha);
}

inline void bind_baddbmm(py::module &m) {

    m.def("baddbmm",
          &ops::py_baddbmm,
          py::arg("input") = py::none(),
          py::arg("mat1"),
          py::arg("mat2"),
          py::arg("beta"),
          py::arg("alpha"),
          R"doc(Performs a batch matrix multiplication of the matrices `mat1` and `mat2`. The matrix `input` is added to the final result.)doc");

    m.def("baddbmm_",
          &ops::py_baddbmm_,
          py::arg("out"),
          py::arg("input") = py::none(),
          py::arg("mat1"),
          py::arg("mat2"),
          py::arg("beta"),
          py::arg("alpha"),
          R"doc(In-place, performs a batch matrix multiplication of the matrices `mat1` and `mat2`. The matrix `input` is added to the final result.)doc");
}

} // namespace infinicore::ops
