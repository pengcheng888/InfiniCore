#include "deepseek_v4_biased_topk_kernel.hpp"

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

namespace infinicore::op::deepseek_v4_biased_topk {
namespace {

constexpr int kGenericBlockSize = 256;
constexpr int kDsv4BlockSize = 256;
#if defined(__HIP_PLATFORM_AMD__)
constexpr int kDeviceWarpSize = 64;
constexpr unsigned long long kFullWarpMask = 0xffffffffffffffffull;
#else
constexpr int kDeviceWarpSize = 32;
constexpr unsigned int kFullWarpMask = 0xffffffffu;
#endif
constexpr int kMaxExperts = 512;
constexpr int kMaxTopK = 16;
constexpr int kDsv4Experts = 256;
constexpr int kDsv4TopK = 6;
constexpr int kDsv4ExpertsPerLane = kDsv4Experts / kDeviceWarpSize;

__device__ __forceinline__ float sqrt_softplus(float x) {
    const float softplus = fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
    return sqrtf(softplus);
}

__device__ __forceinline__ void warp_argmax(float &value, int32_t &expert) {
#pragma unroll
    for (int offset = kDeviceWarpSize / 2; offset > 0; offset >>= 1) {
        const float other_value = __shfl_down_sync(kFullWarpMask, value, offset);
        const int32_t other_expert = __shfl_down_sync(kFullWarpMask, expert, offset);
        if (other_value > value || (other_value == value && other_expert >= 0 && (expert < 0 || other_expert < expert))) {
            value = other_value;
            expert = other_expert;
        }
    }
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = kDeviceWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullWarpMask, value, offset);
    }
    return value;
}

__global__ void biased_topk_kernel(float *__restrict__ topk_weights,
                                   int32_t *__restrict__ topk_indices,
                                   const float *__restrict__ router_logits,
                                   const float *__restrict__ correction_bias,
                                   int64_t tokens,
                                   int64_t num_experts,
                                   int64_t topk,
                                   bool renormalize) {
    const int64_t token = blockIdx.x;
    const int tid = threadIdx.x;
    if (token >= tokens) {
        return;
    }

    __shared__ float choice_scores[kMaxExperts];
    __shared__ float original_scores[kMaxExperts];
    __shared__ float reduce_values[kGenericBlockSize];
    __shared__ int32_t reduce_indices[kGenericBlockSize];
    __shared__ int32_t selected[kMaxTopK];
    __shared__ float selected_weights[kMaxTopK];
    __shared__ float routed_sum;

    for (int64_t expert = tid; expert < num_experts; expert += blockDim.x) {
        const float score = sqrt_softplus(router_logits[token * num_experts + expert]);
        original_scores[expert] = score;
        choice_scores[expert] = score + correction_bias[expert];
    }
    __syncthreads();

    for (int k = 0; k < topk; ++k) {
        float best_value = -FLT_MAX;
        int32_t best_expert = -1;
        for (int64_t expert = tid; expert < num_experts; expert += blockDim.x) {
            const float value = choice_scores[expert];
            if (value > best_value || (value == best_value && static_cast<int32_t>(expert) < best_expert)) {
                best_value = value;
                best_expert = static_cast<int32_t>(expert);
            }
        }

        reduce_values[tid] = best_value;
        reduce_indices[tid] = best_expert;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float other_value = reduce_values[tid + stride];
                const int32_t other_expert = reduce_indices[tid + stride];
                if (other_value > reduce_values[tid] ||
                    (other_value == reduce_values[tid] && other_expert >= 0 &&
                     (reduce_indices[tid] < 0 || other_expert < reduce_indices[tid]))) {
                    reduce_values[tid] = other_value;
                    reduce_indices[tid] = other_expert;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            selected[k] = reduce_indices[0];
            selected_weights[k] = reduce_indices[0] >= 0 ? original_scores[reduce_indices[0]] : 0.0f;
            if (reduce_indices[0] >= 0) {
                choice_scores[reduce_indices[0]] = -FLT_MAX;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        float sum = 0.0f;
        for (int k = 0; k < topk; ++k) {
            sum += selected_weights[k];
        }
        routed_sum = sum;
    }
    __syncthreads();

    if (tid < topk) {
        const int64_t offset = token * topk + tid;
        const float norm = (renormalize && routed_sum > 0.0f) ? routed_sum : 1.0f;
        topk_weights[offset] = selected_weights[tid] / norm;
        topk_indices[offset] = selected[tid];
    }
}

__global__ void biased_topk_dsv4_kernel(float *__restrict__ topk_weights,
                                        int32_t *__restrict__ topk_indices,
                                        const float *__restrict__ router_logits,
                                        const float *__restrict__ correction_bias,
                                        int64_t tokens) {
    const int warp_id = threadIdx.x / kDeviceWarpSize;
    const int lane = threadIdx.x & (kDeviceWarpSize - 1);
    const int warps_per_block = blockDim.x / kDeviceWarpSize;
    const int64_t token = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_id;
    if (token >= tokens) {
        return;
    }

    float choice[kDsv4ExpertsPerLane];
    float original[kDsv4ExpertsPerLane];
#pragma unroll
    for (int i = 0; i < kDsv4ExpertsPerLane; ++i) {
        const int expert = lane + i * kDeviceWarpSize;
        const float score = sqrt_softplus(router_logits[token * kDsv4Experts + expert]);
        original[i] = score;
        choice[i] = score + correction_bias[expert];
    }

    int32_t selected_experts[kDsv4TopK];
    float selected_weights[kDsv4TopK];
    float routed_sum = 0.0f;

#pragma unroll
    for (int k = 0; k < kDsv4TopK; ++k) {
        float best_value = -FLT_MAX;
        int32_t best_expert = -1;
#pragma unroll
        for (int i = 0; i < kDsv4ExpertsPerLane; ++i) {
            const int32_t expert = lane + i * kDeviceWarpSize;
            const float value = choice[i];
            if (value > best_value || (value == best_value && expert < best_expert)) {
                best_value = value;
                best_expert = expert;
            }
        }

        warp_argmax(best_value, best_expert);
        const int32_t global_expert = __shfl_sync(kFullWarpMask, best_expert, 0);

        float selected_weight = 0.0f;
#pragma unroll
        for (int i = 0; i < kDsv4ExpertsPerLane; ++i) {
            const int32_t expert = lane + i * kDeviceWarpSize;
            if (expert == global_expert) {
                selected_weight = original[i];
                choice[i] = -FLT_MAX;
            }
        }
        selected_weight = warp_sum(selected_weight);
        selected_weight = __shfl_sync(kFullWarpMask, selected_weight, 0);
        selected_experts[k] = global_expert;
        selected_weights[k] = selected_weight;
        routed_sum += selected_weight;
    }

    if (lane < kDsv4TopK) {
        const int64_t offset = token * kDsv4TopK + lane;
        topk_weights[offset] = selected_weights[lane] / routed_sum;
        topk_indices[offset] = selected_experts[lane];
    }
}

} // namespace

void launch_biased_topk_generic(float *topk_weights,
                                int32_t *topk_indices,
                                const float *router_logits,
                                const float *correction_bias,
                                int64_t tokens,
                                int64_t num_experts,
                                int64_t topk,
                                bool renormalize,
                                void *stream) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    biased_topk_kernel<<<static_cast<unsigned int>(tokens), kGenericBlockSize, 0, cuda_stream>>>(
        topk_weights,
        topk_indices,
        router_logits,
        correction_bias,
        tokens,
        num_experts,
        topk,
        renormalize);
}

void launch_biased_topk_dsv4(float *topk_weights,
                             int32_t *topk_indices,
                             const float *router_logits,
                             const float *correction_bias,
                             int64_t tokens,
                             void *stream) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const int warps_per_block = kDsv4BlockSize / kDeviceWarpSize;
    const int blocks = static_cast<int>((tokens + warps_per_block - 1) / warps_per_block);
    biased_topk_dsv4_kernel<<<blocks, kDsv4BlockSize, 0, cuda_stream>>>(
        topk_weights,
        topk_indices,
        router_logits,
        correction_bias,
        tokens);
}

void launch_biased_topk(float *topk_weights,
                        int32_t *topk_indices,
                        const float *router_logits,
                        const float *correction_bias,
                        int64_t tokens,
                        int64_t num_experts,
                        int64_t topk,
                        bool renormalize,
                        void *stream) {
    if (num_experts == kDsv4Experts && topk == kDsv4TopK && renormalize) {
        launch_biased_topk_dsv4(topk_weights, topk_indices, router_logits, correction_bias, tokens, stream);
        return;
    }
    launch_biased_topk_generic(topk_weights, topk_indices, router_logits, correction_bias, tokens, num_experts, topk, renormalize, stream);
}

} // namespace infinicore::op::deepseek_v4_biased_topk
