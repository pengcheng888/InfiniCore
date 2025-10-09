#pragma once

#include "op/matmul.hpp"
#include "op/ones.hpp"
#include "op/zeros.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::op {

inline void bind(py::module &m) {
    bind_matmul(m);
    bind_zeros(m);
    bind_ones(m);
}

} // namespace infinicore::op
