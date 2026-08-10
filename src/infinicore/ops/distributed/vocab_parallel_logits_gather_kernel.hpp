#pragma once

#include <cstddef>

namespace infinicore::op::distributed {

void launch_vocab_parallel_logits_reorder(void *output,
                                          const void *gathered,
                                          size_t tokens,
                                          size_t vocab_local,
                                          size_t world_size,
                                          size_t element_size,
                                          void *stream);

} // namespace infinicore::op::distributed
