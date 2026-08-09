#include "deepseek_v4_c4_act_quant_fused_scale_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace infinicore::op::deepseek_v4_c4_act_quant_fused_scale {
namespace {

constexpr int kHeadDim = 128;
constexpr float kFp8Max = 448.0f;

__device__ __forceinline__ float load_scalar(const void *__restrict__ ptr, int64_t idx, int dtype) {
    if (dtype == kDsv4BF16) {
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(ptr)[idx]);
    }
    if (dtype == kDsv4F16) {
        return __half2float(reinterpret_cast<const __half *>(ptr)[idx]);
    }
    return reinterpret_cast<const float *>(ptr)[idx];
}

__device__ __forceinline__ float scale_from_max_abs(float max_abs) {
    return fmaxf(max_abs, 1.0e-4f) * 0.0022321429569274187f;
}

__device__ __forceinline__ uint8_t fp8_e4m3_byte(float value) {
    value = fminf(fmaxf(value, -kFp8Max), kFp8Max);
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

__global__ void c4_act_quant_fused_scale_kernel(const void *__restrict__ q,
                                                int q_dtype,
                                                const void *__restrict__ weights,
                                                int weights_dtype,
                                                uint8_t *__restrict__ q_fp8,
                                                float *__restrict__ q_scale,
                                                float *__restrict__ fused_weights,
                                                int64_t rows,
                                                float weight_scale) {
    const int64_t row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= rows || lane >= kHeadDim) {
        return;
    }

    __shared__ float reduce[kHeadDim];
    __shared__ float scale;
    const float value = load_scalar(q, row * kHeadDim + lane, q_dtype);
    reduce[lane] = fabsf(value);
    __syncthreads();

#pragma unroll
    for (int stride = kHeadDim / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reduce[lane] = fmaxf(reduce[lane], reduce[lane + stride]);
        }
        __syncthreads();
    }

    if (lane == 0) {
        scale = scale_from_max_abs(reduce[0]);
        q_scale[row] = scale;
        fused_weights[row] = load_scalar(weights, row, weights_dtype) * weight_scale * scale;
    }
    __syncthreads();

    q_fp8[row * kHeadDim + lane] = fp8_e4m3_byte(value / scale);
}

} // namespace

void launch_c4_act_quant_fused_scale(const void *q,
                                     int q_dtype,
                                     const void *weights,
                                     int weights_dtype,
                                     uint8_t *q_fp8,
                                     float *q_scale,
                                     float *fused_weights,
                                     int64_t rows,
                                     float weight_scale,
                                     void *stream) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    c4_act_quant_fused_scale_kernel<<<static_cast<unsigned int>(rows), kHeadDim, 0, cuda_stream>>>(
        q, q_dtype, weights, weights_dtype, q_fp8, q_scale, fused_weights, rows, weight_scale);
}

} // namespace infinicore::op::deepseek_v4_c4_act_quant_fused_scale
