#pragma once

#include "../common/op.hpp"

#include <cstddef>

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

void store_flash_mla_bf16_cache_(const Tensor &input,
                                 Tensor cache,
                                 const Tensor &indices,
                                 size_t rope_dim);

} // namespace infinicore::op::deepseek_v4
