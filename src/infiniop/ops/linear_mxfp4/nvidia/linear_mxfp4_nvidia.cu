#include "linear_mxfp4_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../../mxfp4_common/cuda/mxfp4_kernel.cuh"

namespace op::linear_mxfp4::nvidia {
namespace {

template <typename T, size_t M_TILE>
INFINIOP_CUDA_KERNEL linear_mxfp4_kernel(
    T *output,
    const T *input,
    const uint8_t *packed_weight,
    const uint8_t *weight_scale,
    const T *bias,
    size_t M,
    size_t N,
    size_t K,
    float alpha) {
    const size_t n = blockIdx.x;
    const size_t m_begin = blockIdx.y * M_TILE;
    const size_t packed_width = K / 2;
    const size_t scale_width = K / 32;
    const auto *packed_row = packed_weight + n * packed_width;
    const auto *scale_row = weight_scale + n * scale_width;

    float sums[M_TILE] = {};
    for (size_t packed_k = threadIdx.x; packed_k < packed_width; packed_k += blockDim.x) {
        float weight_low;
        float weight_high;
        mxfp4DecodePair(
            packed_row[packed_k], scale_row[packed_k / 16], weight_low, weight_high);
        const size_t k = packed_k * 2;
#pragma unroll
        for (size_t tile_m = 0; tile_m < M_TILE; ++tile_m) {
            const size_t m = m_begin + tile_m;
            if (m < M) {
                const size_t input_offset = m * K + k;
                sums[tile_m] += mxfp4Load(input, input_offset) * weight_low
                              + mxfp4Load(input, input_offset + 1) * weight_high;
            }
        }
    }

    extern __shared__ float scratch[];
    mxfp4BlockReduce(sums, scratch);
    if (threadIdx.x == 0) {
#pragma unroll
        for (size_t tile_m = 0; tile_m < M_TILE; ++tile_m) {
            const size_t m = m_begin + tile_m;
            if (m < M) {
                float value = alpha * sums[tile_m];
                if (bias != nullptr) {
                    value += mxfp4Load(bias, n);
                }
                output[m * N + n] = mxfp4Store<T>(value);
            }
        }
    }
}

template <typename T>
void launch(T *output,
            const T *input,
            const uint8_t *packed_weight,
            const uint8_t *weight_scale,
            const T *bias,
            const LinearMxfp4Info &info,
            cudaStream_t stream) {
    constexpr size_t block_size = 256;
    if (info.M == 1) {
        linear_mxfp4_kernel<T, 1><<<dim3(info.N, 1), block_size,
                                    block_size * sizeof(float), stream>>>(
            output, input, packed_weight, weight_scale, bias,
            info.M, info.N, info.K, info.alpha);
        return;
    }

    constexpr size_t m_tile = 4;
    const dim3 grid(info.N, (info.M + m_tile - 1) / m_tile);
    linear_mxfp4_kernel<T, m_tile><<<grid, block_size,
                                     m_tile * block_size * sizeof(float), stream>>>(
        output, input, packed_weight, weight_scale, bias,
        info.M, info.N, info.K, info.alpha);
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
    infiniopTensorDescriptor_t packed_weight_desc,
    infiniopTensorDescriptor_t weight_scale_desc,
    infiniopTensorDescriptor_t bias_desc,
    float alpha) {
    auto info = LinearMxfp4Info::create(
        output_desc, input_desc, packed_weight_desc, weight_scale_desc, bias_desc, alpha);
    CHECK_RESULT(info);
    auto nvidia_handle = reinterpret_cast<device::nvidia::Handle *>(handle);
    *desc_ptr = new Descriptor(
        new Opaque{nvidia_handle->internal()}, info.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *, size_t,
    void *output,
    const void *input,
    const void *packed_weight,
    const void *weight_scale,
    const void *bias,
    void *stream) const {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const auto *packed_ptr = reinterpret_cast<const uint8_t *>(packed_weight);
    const auto *scale_ptr = reinterpret_cast<const uint8_t *>(weight_scale);
    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        launch(reinterpret_cast<half *>(output),
               reinterpret_cast<const half *>(input),
               packed_ptr, scale_ptr, reinterpret_cast<const half *>(bias),
               _info, cuda_stream);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_BF16:
        launch(reinterpret_cast<__nv_bfloat16 *>(output),
               reinterpret_cast<const __nv_bfloat16 *>(input),
               packed_ptr, scale_ptr, reinterpret_cast<const __nv_bfloat16 *>(bias),
               _info, cuda_stream);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F32:
        launch(reinterpret_cast<float *>(output),
               reinterpret_cast<const float *>(input),
               packed_ptr, scale_ptr, reinterpret_cast<const float *>(bias),
               _info, cuda_stream);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::linear_mxfp4::nvidia
