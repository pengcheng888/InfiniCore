#pragma once

#include "common/op.hpp"

namespace infinicore::op {

void deepseek_v4_dcu_alloc_decode_kernel_(const Tensor &seq_lens,
                                          const Tensor &last_loc,
                                          const Tensor &free_page,
                                          Tensor out_indices,
                                          int bs,
                                          int page_size);

void deepseek_v4_dcu_alloc_extend_kernel_(const Tensor &pre_lens,
                                          const Tensor &seq_lens,
                                          const Tensor &last_loc,
                                          const Tensor &free_page,
                                          Tensor out_indices,
                                          int bs,
                                          int page_size);

} // namespace infinicore::op
