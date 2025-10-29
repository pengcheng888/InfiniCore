#pragma once

#include <pybind11/pybind11.h>

#include "ops/add.hpp"
#include "ops/attention.hpp"
#include "ops/linear.hpp"
#include "ops/matmul.hpp"
#include "ops/rearrange.hpp"
#include "ops/rms_norm.hpp"
#include "ops/rope.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind(py::module &m) {
    bind_add(m);
    bind_attention(m);
    bind_matmul(m);
    bind_rearrange(m);
    bind_rms_norm(m);
    bind_rope(m);
    bind_linear(m);
}

} // namespace infinicore::ops
