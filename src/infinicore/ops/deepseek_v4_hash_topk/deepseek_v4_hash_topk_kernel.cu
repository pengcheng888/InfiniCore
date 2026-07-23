#include "deepseek_v4_hash_topk_kernel.hpp"

#include <cuda_runtime.h>
#include <math.h>

namespace infinicore::op::deepseek_v4_hash_topk {
namespace {

constexpr int kGenericBlockSize = 32;
constexpr int kDsv4BlockSize = 32;
constexpr int kMaxTopK = 32;
constexpr int kDsv4Experts = 256;
constexpr int kDsv4TopK = 6;

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
                                 bool renormalize) {
    const int64_t token = blockIdx.x;
    const int lane = threadIdx.x;
    if (token >= tokens) {
        return;
    }

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

    if (lane < topk) {
        const int64_t offset = token * topk + lane;
        const float norm = (renormalize && routed_sum > 0.0f) ? routed_sum : 1.0f;
        topk_weights[offset] = weights[lane] / norm;
        topk_indices[offset] = experts[lane];
    }
}

template <typename Tid2EidT>
__global__ void hash_topk_dsv4_kernel(float *__restrict__ topk_weights,
                                      int32_t *__restrict__ topk_indices,
                                      const float *__restrict__ router_logits,
                                      const int64_t *__restrict__ input_ids,
                                      const Tid2EidT *__restrict__ tid2eid,
                                      int64_t tokens) {
    const int64_t token = blockIdx.x;
    const int lane = threadIdx.x;
    if (token >= tokens) {
        return;
    }

    __shared__ float weights[kDsv4TopK];
    __shared__ int32_t experts[kDsv4TopK];
    __shared__ float routed_sum;

    if (lane < kDsv4TopK) {
        const int64_t token_id = input_ids[token];
        const int32_t expert_id = static_cast<int32_t>(tid2eid[token_id * kDsv4TopK + lane]);
        experts[lane] = expert_id;
        weights[lane] = sqrt_softplus(router_logits[token * kDsv4Experts + expert_id]);
    }
    __syncthreads();

    if (lane == 0) {
        routed_sum = weights[0] + weights[1] + weights[2] + weights[3] + weights[4] + weights[5];
    }
    __syncthreads();

    if (lane < kDsv4TopK) {
        const int64_t offset = token * kDsv4TopK + lane;
        topk_weights[offset] = weights[lane] / routed_sum;
        topk_indices[offset] = experts[lane];
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
                                bool renormalize,
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
        renormalize);
}

template <typename Tid2EidT>
void launch_hash_topk_dsv4_t(float *topk_weights,
                             int32_t *topk_indices,
                             const float *router_logits,
                             const int64_t *input_ids,
                             const void *tid2eid,
                             int64_t tokens,
                             cudaStream_t stream) {
    hash_topk_dsv4_kernel<Tid2EidT><<<static_cast<unsigned int>(tokens), kDsv4BlockSize, 0, stream>>>(
        topk_weights,
        topk_indices,
        router_logits,
        input_ids,
        reinterpret_cast<const Tid2EidT *>(tid2eid),
        tokens);
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
                              bool renormalize,
                              void *stream) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (tid2eid_i64) {
        launch_hash_topk_generic_t<int64_t>(
            topk_weights, topk_indices, router_logits, input_ids, tid2eid, tokens, num_experts, topk, renormalize, cuda_stream);
    } else {
        launch_hash_topk_generic_t<int32_t>(
            topk_weights, topk_indices, router_logits, input_ids, tid2eid, tokens, num_experts, topk, renormalize, cuda_stream);
    }
}

void launch_hash_topk_dsv4(float *topk_weights,
                           int32_t *topk_indices,
                           const float *router_logits,
                           const int64_t *input_ids,
                           const void *tid2eid,
                           bool tid2eid_i64,
                           int64_t tokens,
                           void *stream) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (tid2eid_i64) {
        launch_hash_topk_dsv4_t<int64_t>(topk_weights, topk_indices, router_logits, input_ids, tid2eid, tokens, cuda_stream);
    } else {
        launch_hash_topk_dsv4_t<int32_t>(topk_weights, topk_indices, router_logits, input_ids, tid2eid, tokens, cuda_stream);
    }
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
                      bool renormalize,
                      void *stream) {
    if (num_experts == kDsv4Experts && topk == kDsv4TopK && renormalize) {
        launch_hash_topk_dsv4(topk_weights, topk_indices, router_logits, input_ids, tid2eid, tid2eid_i64, tokens, stream);
        return;
    }
    launch_hash_topk_generic(
        topk_weights, topk_indices, router_logits, input_ids, tid2eid, tid2eid_i64, tokens, num_experts, topk, renormalize, stream);
}

} // namespace infinicore::op::deepseek_v4_hash_topk
