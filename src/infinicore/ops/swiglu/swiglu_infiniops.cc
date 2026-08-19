#include "infinicore/ops/swiglu.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

namespace infinicore::op::swiglu_impl::infiniop {
void *plan(Tensor c, const Tensor &a, const Tensor &b);
void run(void *planned_meta);
void cleanup(void **planned_meta_ptr);
} // namespace infinicore::op::swiglu_impl::infiniop

namespace infinicore::op::swiglu_impl::infiniops {

static bool registered = []() {
    // Reuse the shared InfiniOp elementwise implementation in InfiniOps-enabled builds.
    ::infinicore::op::infiniops::registerSupportedDevices(SwiGLU::plan_dispatcher(), &infiniop::plan);
    ::infinicore::op::infiniops::registerSupportedDevices(SwiGLU::run_dispatcher(), &infiniop::run);
    ::infinicore::op::infiniops::registerSupportedDevices(SwiGLU::cleanup_dispatcher(), &infiniop::cleanup);
    return true;
}();

} // namespace infinicore::op::swiglu_impl::infiniops
#endif
