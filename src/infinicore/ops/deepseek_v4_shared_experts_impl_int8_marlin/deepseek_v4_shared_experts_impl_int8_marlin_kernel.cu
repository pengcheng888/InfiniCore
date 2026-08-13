#include "deepseek_v4_shared_experts_impl_int8_marlin_kernel.hpp"

#include <cuda_runtime.h>
#include <stdint.h>

namespace infinicore::op::deepseek_v4_shared_experts_impl_int8_marlin {
namespace {

constexpr int kThreads = 256;

__global__ void fill_single_expert_metadata_kernel(int32_t *__restrict__ sorted_token_ids,
                                                   int32_t *__restrict__ expert_ids,
                                                   int32_t *__restrict__ num_tokens_post_pad,
                                                   float *__restrict__ topk_weights,
                                                   int64_t tokens,
                                                   int32_t top_k,
                                                   int64_t flat_topk,
                                                   int32_t padded_tokens,
                                                   int32_t num_blocks) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < padded_tokens) {
        sorted_token_ids[idx] = idx < flat_topk ? static_cast<int32_t>(idx) : static_cast<int32_t>(flat_topk);
    }
    if (idx < flat_topk) {
        topk_weights[idx] = (idx % top_k) == 0 ? 1.0f : 0.0f;
    }
    if (idx < num_blocks) {
        expert_ids[idx] = 0;
    }
    if (idx == 0) {
        num_tokens_post_pad[0] = padded_tokens;
    }
    return;
}

} // namespace

void launch_fill_single_expert_metadata(void *sorted_token_ids,
                                        void *expert_ids,
                                        void *num_tokens_post_pad,
                                        void *topk_weights,
                                        int64_t tokens,
                                        int top_k,
                                        int block_size,
                                        void *stream) {
    const int64_t flat_topk = tokens * top_k;
    const int64_t padded = ((flat_topk + block_size - 1 + block_size - 1) / block_size) * block_size;
    const int64_t num_blocks = padded / block_size;
    const int64_t total = padded > num_blocks ? padded : num_blocks;
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    fill_single_expert_metadata_kernel<<<blocks, kThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<int32_t *>(sorted_token_ids),
        reinterpret_cast<int32_t *>(expert_ids),
        reinterpret_cast<int32_t *>(num_tokens_post_pad),
        reinterpret_cast<float *>(topk_weights),
        tokens,
        top_k,
        flat_topk,
        static_cast<int32_t>(padded),
        static_cast<int32_t>(num_blocks));
    return;
}

} // namespace infinicore::op::deepseek_v4_shared_experts_impl_int8_marlin
