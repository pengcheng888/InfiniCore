#include "fused_moe_mxfp4_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../../mxfp4_common/cuda/mxfp4_kernel.cuh"

namespace op::fused_moe_mxfp4::nvidia {
namespace {

__device__ __forceinline__ float activate(float gate,
                                          float up,
                                          infiniopFusedMoeActivation_t activation) {
    if (activation == INFINIOP_FUSED_MOE_ACT_SITUGLU) {
        constexpr float beta = 4.0f;
        constexpr float linear_beta = 25.0f;
        const float situ_gate = beta * tanhf(gate / beta) / (1.0f + expf(-gate));
        const float bounded_up = linear_beta * tanhf(up / linear_beta);
        return situ_gate * bounded_up;
    }
    return gate / (1.0f + expf(-gate)) * up;
}

template <typename T>
INFINIOP_CUDA_KERNEL fused_moe_mxfp4_w13_kernel(
    T *activated,
    const T *input,
    const int32_t *selected_experts,
    const uint8_t *w13_packed,
    const uint8_t *w13_scale,
    size_t route_count,
    size_t topk,
    size_t num_experts,
    size_t hidden_size,
    size_t intermediate_size,
    infiniopFusedMoeActivation_t activation) {
    const size_t block = blockIdx.x;
    const size_t route = block / intermediate_size;
    const size_t i = block - route * intermediate_size;
    if (route >= route_count || i >= intermediate_size) {
        return;
    }

    const int32_t expert = selected_experts[route];
    if (expert < 0 || static_cast<size_t>(expert) >= num_experts) {
        if (threadIdx.x == 0) {
            activated[route * intermediate_size + i] = mxfp4Store<T>(0.0f);
        }
        return;
    }

    const size_t token = route / topk;
    const size_t packed_width = hidden_size / 2;
    const size_t scale_width = hidden_size / 32;
    const size_t gate_row = (static_cast<size_t>(expert) * 2 * intermediate_size + i);
    const size_t up_row = gate_row + intermediate_size;
    const auto *gate_packed = w13_packed + gate_row * packed_width;
    const auto *gate_scale = w13_scale + gate_row * scale_width;
    const auto *up_packed = w13_packed + up_row * packed_width;
    const auto *up_scale = w13_scale + up_row * scale_width;
    const auto *token_input = input + token * hidden_size;

    float sums[2] = {};
    for (size_t packed_k = threadIdx.x; packed_k < packed_width; packed_k += blockDim.x) {
        float gate_low;
        float gate_high;
        float up_low;
        float up_high;
        mxfp4DecodePair(
            gate_packed[packed_k], gate_scale[packed_k / 16], gate_low, gate_high);
        mxfp4DecodePair(
            up_packed[packed_k], up_scale[packed_k / 16], up_low, up_high);
        const size_t k = packed_k * 2;
        const float input_low = mxfp4Load(token_input, k);
        const float input_high = mxfp4Load(token_input, k + 1);
        sums[0] += input_low * gate_low + input_high * gate_high;
        sums[1] += input_low * up_low + input_high * up_high;
    }

    extern __shared__ float scratch[];
    mxfp4BlockReduce(sums, scratch);
    if (threadIdx.x == 0) {
        activated[route * intermediate_size + i]
            = mxfp4Store<T>(activate(sums[0], sums[1], activation));
    }
}

template <typename T>
INFINIOP_CUDA_KERNEL fused_moe_mxfp4_w2_kernel(
    T *output,
    const T *activated,
    const int32_t *selected_experts,
    const float *routing_weights,
    const uint8_t *w2_packed,
    const uint8_t *w2_scale,
    size_t num_tokens,
    size_t topk,
    size_t num_experts,
    size_t hidden_size,
    size_t intermediate_size) {
    const size_t block = blockIdx.x;
    const size_t token = block / hidden_size;
    const size_t h = block - token * hidden_size;
    if (token >= num_tokens || h >= hidden_size) {
        return;
    }

    const size_t packed_width = intermediate_size / 2;
    const size_t scale_width = intermediate_size / 32;
    float output_value = 0.0f;
    extern __shared__ float scratch[];
    for (size_t route_index = 0; route_index < topk; ++route_index) {
        const size_t route = token * topk + route_index;
        const int32_t expert = selected_experts[route];
        if (expert < 0 || static_cast<size_t>(expert) >= num_experts) {
            continue;
        }
        const size_t weight_row = static_cast<size_t>(expert) * hidden_size + h;
        const auto *packed_row = w2_packed + weight_row * packed_width;
        const auto *scale_row = w2_scale + weight_row * scale_width;
        const auto *route_input = activated + route * intermediate_size;

        float sum[1] = {};
        for (size_t packed_k = threadIdx.x; packed_k < packed_width; packed_k += blockDim.x) {
            float weight_low;
            float weight_high;
            mxfp4DecodePair(
                packed_row[packed_k], scale_row[packed_k / 16], weight_low, weight_high);
            const size_t k = packed_k * 2;
            sum[0] += mxfp4Load(route_input, k) * weight_low
                    + mxfp4Load(route_input, k + 1) * weight_high;
        }
        mxfp4BlockReduce(sum, scratch);
        if (threadIdx.x == 0) {
            output_value += routing_weights[route] * sum[0];
        }
    }
    if (threadIdx.x == 0) {
        output[token * hidden_size + h] = mxfp4Store<T>(output_value);
    }
}

template <typename T>
void launch(T *output,
            T *activated,
            const T *input,
            const int32_t *selected_experts,
            const float *routing_weights,
            const uint8_t *w13_packed,
            const uint8_t *w13_scale,
            const uint8_t *w2_packed,
            const uint8_t *w2_scale,
            const FusedMoeMxfp4Info &info,
            cudaStream_t stream) {
    constexpr size_t block_size = 256;
    const size_t w13_grid = info.intermediate_size * info.routeCount();
    fused_moe_mxfp4_w13_kernel<<<w13_grid, block_size,
                                 2 * block_size * sizeof(float), stream>>>(
        activated, input, selected_experts, w13_packed, w13_scale,
        info.routeCount(), info.topk, info.num_experts,
        info.hidden_size, info.intermediate_size, info.activation);

    const size_t w2_grid = info.hidden_size * info.num_tokens;
    fused_moe_mxfp4_w2_kernel<<<w2_grid, block_size,
                                block_size * sizeof(float), stream>>>(
        output, activated, selected_experts, routing_weights, w2_packed, w2_scale,
        info.num_tokens, info.topk, info.num_experts,
        info.hidden_size, info.intermediate_size);
}

size_t dtype_size(infiniDtype_t dtype) {
    return dtype == INFINI_DTYPE_F32 ? sizeof(float) : sizeof(uint16_t);
}

} // namespace

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t selected_experts_desc,
    infiniopTensorDescriptor_t routing_weights_desc,
    infiniopTensorDescriptor_t w13_packed_desc,
    infiniopTensorDescriptor_t w13_scale_desc,
    infiniopTensorDescriptor_t w2_packed_desc,
    infiniopTensorDescriptor_t w2_scale_desc,
    infiniopFusedMoeActivation_t activation) {
    auto info = FusedMoeMxfp4Info::create(
        output_desc, input_desc, selected_experts_desc, routing_weights_desc,
        w13_packed_desc, w13_scale_desc, w2_packed_desc, w2_scale_desc, activation);
    CHECK_RESULT(info);
    auto value = info.take();
    const size_t workspace_size = value.routeCount() * value.intermediate_size * dtype_size(value.dtype);
    auto nvidia_handle = reinterpret_cast<device::nvidia::Handle *>(handle);
    *desc_ptr = new Descriptor(
        new Opaque{nvidia_handle->internal()}, value, workspace_size,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *selected_experts,
    const void *routing_weights,
    const void *w13_packed,
    const void *w13_scale,
    const void *w2_packed,
    const void *w2_scale,
    void *stream) const {
    CHECK_OR_RETURN(workspace != nullptr && workspace_size >= _workspace_size,
                    INFINI_STATUS_INSUFFICIENT_WORKSPACE);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const auto *ids = reinterpret_cast<const int32_t *>(selected_experts);
    const auto *weights = reinterpret_cast<const float *>(routing_weights);
    const auto *w13 = reinterpret_cast<const uint8_t *>(w13_packed);
    const auto *w13_s = reinterpret_cast<const uint8_t *>(w13_scale);
    const auto *w2 = reinterpret_cast<const uint8_t *>(w2_packed);
    const auto *w2_s = reinterpret_cast<const uint8_t *>(w2_scale);
    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        launch(reinterpret_cast<half *>(output),
               reinterpret_cast<half *>(workspace),
               reinterpret_cast<const half *>(input),
               ids, weights, w13, w13_s, w2, w2_s, _info, cuda_stream);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_BF16:
        launch(reinterpret_cast<__nv_bfloat16 *>(output),
               reinterpret_cast<__nv_bfloat16 *>(workspace),
               reinterpret_cast<const __nv_bfloat16 *>(input),
               ids, weights, w13, w13_s, w2, w2_s, _info, cuda_stream);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F32:
        launch(reinterpret_cast<float *>(output),
               reinterpret_cast<float *>(workspace),
               reinterpret_cast<const float *>(input),
               ids, weights, w13, w13_s, w2, w2_s, _info, cuda_stream);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::fused_moe_mxfp4::nvidia
