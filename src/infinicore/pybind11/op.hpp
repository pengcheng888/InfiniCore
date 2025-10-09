#pragma once

#include <pybind11/pybind11.h>

#include "op/matmul.hpp"
#include "op/ones.hpp"
#include "op/zeros.hpp"

namespace py = pybind11;

namespace infinicore::op {

inline void bind(py::module &m) {
    bind_matmul(m);
    bind_ones(m);
    bind_zeros(m);
}

} // namespace infinicore::op
