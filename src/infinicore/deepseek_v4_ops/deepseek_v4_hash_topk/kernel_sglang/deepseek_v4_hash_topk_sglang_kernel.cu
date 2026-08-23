#include "../kernel/deepseek_v4_hash_topk_kernel.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>

namespace infinicore::op::deepseek_v4_hash_topk {
namespace {

constexpr int kBlockSize = 128;
#if defined(__HIP_PLATFORM_AMD__)
constexpr int kDeviceWarpSize = 64;
constexpr unsigned long long kFullWarpMask = 0xffffffffffffffffull;
#else
constexpr int kDeviceWarpSize = 32;
constexpr unsigned int kFullWarpMask = 0xffffffffu;
#endif

#if defined(__CUDACC__) && !defined(__HIP_PLATFORM_AMD__) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12)
#define INFINICORE_DSV4_GRID_CONSTANT __grid_constant__
#else
#define INFINICORE_DSV4_GRID_CONSTANT
#endif

__device__ __forceinline__ float sqrt_softplus(float x) {
    const float softplus = fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
    return sqrtf(softplus);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = kDeviceWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullWarpMask, value, offset);
    }
    return __shfl_sync(kFullWarpMask, value, 0);
}

template <typename Tid2EidT>
struct MoEHashTopKParams {
    const float *__restrict__ router_logits;
    const int64_t *__restrict__ input_ids;
    const Tid2EidT *__restrict__ tid2eid;
    int32_t *__restrict__ topk_indices;
    float *__restrict__ topk_weights;
    uint32_t tokens;
    uint32_t topk;
    uint32_t num_experts;
    uint32_t num_fused_shared_experts;
    float routed_scaling_factor;
};

template <typename Tid2EidT>
__global__ void hash_topk_sglang_kernel(const MoEHashTopKParams<Tid2EidT> INFINICORE_DSV4_GRID_CONSTANT params) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t token = tid / kDeviceWarpSize;
    const uint32_t lane = tid % kDeviceWarpSize;
    if (token >= params.tokens) {
        return;
    }

    float routed_weight = 0.0f;
    int32_t expert_id = 0;
    if (lane < params.topk) {
        const int64_t token_id = params.input_ids[token];
        expert_id = static_cast<int32_t>(params.tid2eid[token_id * params.topk + lane]);
        routed_weight = sqrt_softplus(params.router_logits[token * params.num_experts + expert_id]);
    }

    const float routed_sum = warp_sum(routed_weight);
    const uint32_t topk_fused = params.topk + params.num_fused_shared_experts;
    if (lane < topk_fused) {
        const bool is_shared = lane >= params.topk;
        const uint32_t offset = token * topk_fused + lane;
        params.topk_indices[offset] = is_shared ? static_cast<int32_t>(params.num_experts + lane - params.topk) : expert_id;
        params.topk_weights[offset] = is_shared ? (1.0f / params.routed_scaling_factor) : (routed_weight / routed_sum);
    }
}

template <typename Tid2EidT>
void launch_hash_topk_sglang_t(float *topk_weights,
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
    constexpr int kWarpsPerBlock = kBlockSize / kDeviceWarpSize;
    const int64_t blocks = (tokens + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const MoEHashTopKParams<Tid2EidT> params{
        router_logits,
        input_ids,
        reinterpret_cast<const Tid2EidT *>(tid2eid),
        topk_indices,
        topk_weights,
        static_cast<uint32_t>(tokens),
        static_cast<uint32_t>(topk),
        static_cast<uint32_t>(num_experts),
        static_cast<uint32_t>(num_fused_shared_experts),
        routed_scaling_factor};
    hash_topk_sglang_kernel<Tid2EidT><<<static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(params);
    return;
}

} // namespace

void launch_hash_topk_sglang(float *topk_weights,
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
        launch_hash_topk_sglang_t<int64_t>(
            topk_weights, topk_indices, router_logits, input_ids, tid2eid, tokens, num_experts, topk, num_fused_shared_experts, routed_scaling_factor, cuda_stream);
    } else {
        launch_hash_topk_sglang_t<int32_t>(
            topk_weights, topk_indices, router_logits, input_ids, tid2eid, tokens, num_experts, topk, num_fused_shared_experts, routed_scaling_factor, cuda_stream);
    }
    return;
}

} // namespace infinicore::op::deepseek_v4_hash_topk
