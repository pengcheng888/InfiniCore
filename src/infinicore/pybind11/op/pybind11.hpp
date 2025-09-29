#pragma once

#include <pybind11/pybind11.h>

#include "matmul/matmul.hpp"
#include "ones/ones.hpp"
#include "rearrange/rearrange.hpp"
// #include "conv/conv.hpp"

namespace infinicore::py::op {

inline void bind(pybind11::module &m) {
    bind_matmul(m);
    bind_rearrange(m);
    bind_ones(m);
    // bind_conv(m);
}

} // namespace infinicore::py::op
