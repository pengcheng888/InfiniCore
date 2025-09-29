#pragma once
#include "../../tensor/tensor.hpp"
#include "infinicore.hpp"
#include <pybind11/pybind11.h>

namespace infinicore::py {

Tensor rearrange(const Tensor &x);
void rearrange_(Tensor &y, const Tensor &x);

inline void bind_rearrange(pybind11::module &m) {
    m.def("rearrange",
          &rearrange,
          pybind11::arg("x"),
          R"doc(Rearrange tensor dimensions.)doc");

    m.def("rearrange_",
          &rearrange_,
          pybind11::arg("y"),
          pybind11::arg("x"),
          R"doc(In-place tensor rearrangement.)doc");
}

} // namespace infinicore::py
