#include "infinicore/ops/deepseek_v4_mhc_post.hpp"

namespace infinicore::op {

void deepseek_v4_mhc_post_(Tensor y,
                           const Tensor &x,
                           const Tensor &residual,
                           const Tensor &post,
                           const Tensor &comb) {
    deepseek_v4_mhc_post_kernel_(y, x, residual, post, comb);
    return;
}

} // namespace infinicore::op
