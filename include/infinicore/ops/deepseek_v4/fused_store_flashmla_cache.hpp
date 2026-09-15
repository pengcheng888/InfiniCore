#pragma once

#include "../common/op.hpp"

namespace infinicore::op::deepseek_v4 {

// NOTE: The InfiniLM/InfiniCore path does not yet cover all functionality of
// SGLang's kernel, including F16 RoPE conversion and F32 C4/C128 inputs.
void fused_store_flashmla_cache_(const Tensor &input,
                                 Tensor cache,
                                 const Tensor &indices,
                                 int page_size);

void fused_store_flashmla_cache_aten_(const Tensor &input,
                                      Tensor cache,
                                      const Tensor &indices,
                                      int page_size);

void fused_store_flashmla_cache_kernel_(const Tensor &input,
                                        Tensor cache,
                                        const Tensor &indices,
                                        int page_size);

} // namespace infinicore::op::deepseek_v4
