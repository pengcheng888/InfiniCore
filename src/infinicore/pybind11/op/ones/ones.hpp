#pragma once

#include "../../tensor/tensor.hpp"
#include "infinicore.hpp"
#include <pybind11/pybind11.h>

namespace infinicore::py {

// Tensor ones(const Tensor &x);
void ones_(Tensor &y, const Tensor &x);

inline void bind_ones(pybind11::module &m) {
    //     m.def("ones",
    //           &infinicore::py::ones,
    //           pybind11::arg("x"));

    m.def("ones_",
          &ones_,
          pybind11::arg("y"),
          pybind11::arg("x"));
}

} // namespace infinicore::py
