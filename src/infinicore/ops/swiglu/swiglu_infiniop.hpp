#pragma once

#include "infinicore/tensor.hpp"

namespace infinicore::op::swiglu_impl::infiniop {

void *plan(Tensor c, const Tensor &a, const Tensor &b);
void run(void *planned_meta);
void cleanup(void **planned_meta_ptr);

} // namespace infinicore::op::swiglu_impl::infiniop
