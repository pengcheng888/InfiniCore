#include "infinicore/ops/deepseek_v4/mhc_post.hpp"

namespace infinicore::op::deepseek_v4 {

void mhc_post_(Tensor y,
               const Tensor &x,
               const Tensor &residual,
               const Tensor &post,
               const Tensor &comb) {
    mhc_post_kernel_(y, x, residual, post, comb);
    return;
}

} // namespace infinicore::op::deepseek_v4
