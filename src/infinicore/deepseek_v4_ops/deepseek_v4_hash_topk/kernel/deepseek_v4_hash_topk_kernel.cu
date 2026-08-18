#include "deepseek_v4_hash_topk_kernel.hpp"

#include <cuda_runtime.h>
#include <math.h>

namespace infinicore::op::deepseek_v4_hash_topk {
namespace {

constexpr int kGenericBlockSize = 32;
constexpr int kNumExperts256TopK6BlockSize = 32;
constexpr int kMaxTopK = 32;
constexpr int kNumExperts256 = 256;
constexpr int kTopK6 = 6;

__device__ __forceinline__ float sqrt_softplus(float x) {
    const float softplus = fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
    return sqrtf(softplus);
}

template <typename Tid2EidT>
__global__ void hash_topk_kernel(float *__restrict__ topk_weights,
                                 int32_t *__restrict__ topk_indices,
                                 const float *__restrict__ router_logits,
                                 const int64_t *__restrict__ input_ids,
                                 const Tid2EidT *__restrict__ tid2eid,
                                 int64_t tokens,
                                 int64_t num_experts,
                                 int64_t topk,
                                 int64_t num_fused_shared_experts,
                                 float routed_scaling_factor) {
    const int64_t token = blockIdx.x;
    const int lane = threadIdx.x;
    if (token >= tokens) {
        return;
    }
    const int64_t topk_fused = topk + num_fused_shared_experts;

    __shared__ float weights[kMaxTopK];
    __shared__ int32_t experts[kMaxTopK];
    __shared__ float routed_sum;

    if (lane < topk) {
        const int64_t token_id = input_ids[token];
        const int32_t expert_id = static_cast<int32_t>(tid2eid[token_id * topk + lane]);
        experts[lane] = expert_id;
        weights[lane] = sqrt_softplus(router_logits[token * num_experts + expert_id]);
    }
    __syncthreads();

    if (lane == 0) {
        float sum = 0.0f;
        for (int i = 0; i < topk; ++i) {
            sum += weights[i];
        }
        routed_sum = sum;
    }
    __syncthreads();

    if (lane < topk_fused) {
        const int64_t offset = token * topk_fused + lane;
        const bool is_shared = lane >= topk;
        topk_weights[offset] = is_shared ? (1.0f / routed_scaling_factor) : (weights[lane] / routed_sum);
        topk_indices[offset] = is_shared ? static_cast<int32_t>(num_experts + lane - topk) : experts[lane];
    }
}

template <typename Tid2EidT>
__global__ void hash_topk_num_experts_256_topk_6_kernel(float *__restrict__ topk_weights,
                                                        int32_t *__restrict__ topk_indices,
                                                        const float *__restrict__ router_logits,
                                                        const int64_t *__restrict__ input_ids,
                                                        const Tid2EidT *__restrict__ tid2eid,
                                                        int64_t tokens,
                                                        int64_t num_fused_shared_experts,
                                                        float routed_scaling_factor) {
    const int64_t token = blockIdx.x;
    const int lane = threadIdx.x;
    if (token >= tokens) {
        return;
    }
    const int64_t topk_fused = kTopK6 + num_fused_shared_experts;

    __shared__ float weights[kTopK6];
    __shared__ int32_t experts[kTopK6];
    __shared__ float routed_sum;

    if (lane < kTopK6) {
        const int64_t token_id = input_ids[token];
        const int32_t expert_id = static_cast<int32_t>(tid2eid[token_id * kTopK6 + lane]);
        experts[lane] = expert_id;
        weights[lane] = sqrt_softplus(router_logits[token * kNumExperts256 + expert_id]);
    }
    __syncthreads();

    if (lane == 0) {
        routed_sum = weights[0] + weights[1] + weights[2] + weights[3] + weights[4] + weights[5];
    }
    __syncthreads();

    if (lane < topk_fused) {
        const int64_t offset = token * topk_fused + lane;
        const bool is_shared = lane >= kTopK6;
        topk_weights[offset] = is_shared ? (1.0f / routed_scaling_factor) : (weights[lane] / routed_sum);
        topk_indices[offset] = is_shared ? static_cast<int32_t>(kNumExperts256 + lane - kTopK6) : experts[lane];
    }
}

template <typename Tid2EidT>
void launch_hash_topk_generic_t(float *topk_weights,
                                int32_t *topk_indices,
                                const float *router_logits,
                                const int64_t *input_ids,
                                const void *tid2eid,
                                int64_t tokens,
                                int64_t num_experts,
                                int64_t topk,
                                int64_t num_fused_shared_experts,
                                float routed_scaling_factor,
                                cudaStream_t stream) {
    hash_topk_kernel<Tid2EidT><<<static_cast<unsigned int>(tokens), kGenericBlockSize, 0, stream>>>(
        topk_weights,
        topk_indices,
        router_logits,
        input_ids,
        reinterpret_cast<const Tid2EidT *>(tid2eid),
        tokens,
        num_experts,
        topk,
        num_fused_shared_experts,
        routed_scaling_factor);
    return;
}

template <typename Tid2EidT>
void launch_hash_topk_num_experts_256_topk_6_t(float *topk_weights,
                                               int32_t *topk_indices,
                                               const float *router_logits,
                                               const int64_t *input_ids,
                                               const void *tid2eid,
                                               int64_t tokens,
                                               int64_t num_fused_shared_experts,
                                               float routed_scaling_factor,
                                               cudaStream_t stream) {
    hash_topk_num_experts_256_topk_6_kernel<Tid2EidT><<<static_cast<unsigned int>(tokens), kNumExperts256TopK6BlockSize, 0, stream>>>(
        topk_weights,
        topk_indices,
        router_logits,
        input_ids,
        reinterpret_cast<const Tid2EidT *>(tid2eid),
        tokens,
        num_fused_shared_experts,
        routed_scaling_factor);
    return;
}

} // namespace

void launch_hash_topk_generic(float *topk_weights,
                              int32_t *topk_indices,
                              const float *router_logits,
                              const int64_t *input_ids,
                              const void *tid2eid,
                              bool tid2eid_i64,
                              int64_t tokens,
                              int64_t num_experts,
                              int64_t topk,
                              int64_t num_fused_shared_experts,
                              float routed_scaling_factor,
                              void *stream) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (tid2eid_i64) {
        launch_hash_topk_generic_t<int64_t>(
            topk_weights, topk_indices, router_logits, input_ids, tid2eid, tokens, num_experts, topk, num_fused_shared_experts, routed_scaling_factor, cuda_stream);
    } else {
        launch_hash_topk_generic_t<int32_t>(
            topk_weights, topk_indices, router_logits, input_ids, tid2eid, tokens, num_experts, topk, num_fused_shared_experts, routed_scaling_factor, cuda_stream);
    }
    return;
}

void launch_hash_topk_num_experts_256_topk_6_(float *topk_weights,
                                              int32_t *topk_indices,
                                              const float *router_logits,
                                              const int64_t *input_ids,
                                              const void *tid2eid,
                                              bool tid2eid_i64,
                                              int64_t tokens,
                                              int64_t num_fused_shared_experts,
                                              float routed_scaling_factor,
                                              void *stream) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (tid2eid_i64) {
        launch_hash_topk_num_experts_256_topk_6_t<int64_t>(topk_weights, topk_indices, router_logits,
                                                           input_ids, tid2eid, tokens, num_fused_shared_experts, routed_scaling_factor, cuda_stream);
    } else {
        launch_hash_topk_num_experts_256_topk_6_t<int32_t>(topk_weights, topk_indices, router_logits,
                                                           input_ids, tid2eid, tokens, num_fused_shared_experts, routed_scaling_factor, cuda_stream);
    }
    return;
}

void launch_hash_topk(float *topk_weights,
                      int32_t *topk_indices,
                      const float *router_logits,
                      const int64_t *input_ids,
                      const void *tid2eid,
                      bool tid2eid_i64,
                      int64_t tokens,
                      int64_t num_experts,
                      int64_t topk,
                      int64_t num_fused_shared_experts,
                      float routed_scaling_factor,
                      void *stream) {
    if (num_experts == kNumExperts256 && topk == kTopK6) {
        launch_hash_topk_num_experts_256_topk_6_(topk_weights, topk_indices, router_logits, input_ids, tid2eid, tid2eid_i64, tokens, num_fused_shared_experts, routed_scaling_factor, stream);
        return;
    }
    launch_hash_topk_generic(
        topk_weights, topk_indices, router_logits, input_ids, tid2eid, tid2eid_i64, tokens, num_experts, topk, num_fused_shared_experts, routed_scaling_factor, stream);
}

} // namespace infinicore::op::deepseek_v4_hash_topk
