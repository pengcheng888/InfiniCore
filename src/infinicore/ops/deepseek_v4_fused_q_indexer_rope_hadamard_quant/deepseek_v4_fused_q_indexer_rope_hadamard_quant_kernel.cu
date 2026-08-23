#include "deepseek_v4_fused_q_indexer_rope_hadamard_quant_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace infinicore::op::deepseek_v4_fused_q_indexer_rope_hadamard_quant {
namespace {

constexpr int kHeadDim = 128;
constexpr int kRopeDim = 64;
constexpr float kFp8Max = 448.0f;
constexpr float kHadamardScale = 0.08838834764831845f;

__device__ __forceinline__ float load_scalar(const void *__restrict__ ptr, int64_t idx, int dtype) {
    if (dtype == kDsv4BF16) {
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(ptr)[idx]);
    }
    if (dtype == kDsv4F16) {
        return __half2float(reinterpret_cast<const __half *>(ptr)[idx]);
    }
    return reinterpret_cast<const float *>(ptr)[idx];
}

__device__ __forceinline__ float roundtrip_scalar(float value, int dtype) {
    if (dtype == kDsv4BF16) {
        return __bfloat162float(__float2bfloat16(value));
    }
    if (dtype == kDsv4F16) {
        return __half2float(__float2half(value));
    }
    return value;
}

__device__ __forceinline__ int64_t load_index(const void *__restrict__ ptr, int64_t idx, bool i64) {
    return i64 ? reinterpret_cast<const int64_t *>(ptr)[idx]
               : static_cast<int64_t>(reinterpret_cast<const int32_t *>(ptr)[idx]);
}

__device__ __forceinline__ float scale_from_max_abs(float max_abs) {
    return fmaxf(max_abs, 1.0e-4f) * 0.0022321429569274187f;
}

__device__ __forceinline__ uint8_t fp8_e4m3_byte(float value) {
    value = fminf(fmaxf(value, -kFp8Max), kFp8Max);
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

__global__ void fused_q_indexer_rope_hadamard_quant_kernel(const void *__restrict__ q,
                                                           int q_dtype,
                                                           const void *__restrict__ weights,
                                                           int weights_dtype,
                                                           uint8_t *__restrict__ q_fp8,
                                                           float *__restrict__ q_scale,
                                                           float *__restrict__ fused_weights,
                                                           float weight_scale,
                                                           const float *__restrict__ freqs_cis,
                                                           const void *__restrict__ positions,
                                                           bool positions_i64,
                                                           int64_t rows,
                                                           int64_t heads) {
    const int64_t row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= rows || lane >= kHeadDim) {
        return;
    }

    __shared__ float values[kHeadDim];
    __shared__ float reduce[kHeadDim];
    __shared__ float scale;

    float value = load_scalar(q, row * kHeadDim + lane, q_dtype);
    const int64_t token = row / heads;
    const int64_t pos = load_index(positions, token, positions_i64);

    if (lane >= kHeadDim - kRopeDim) {
        const int rope_lane = lane - (kHeadDim - kRopeDim);
        const int pair = rope_lane >> 1;
        const int real_lane = kHeadDim - kRopeDim + 2 * pair;
        const int imag_lane = real_lane + 1;
        const float xr = load_scalar(q, row * kHeadDim + real_lane, q_dtype);
        const float xi = load_scalar(q, row * kHeadDim + imag_lane, q_dtype);
        const float c = freqs_cis[pos * kRopeDim + 2 * pair];
        const float s = freqs_cis[pos * kRopeDim + 2 * pair + 1];
        value = (lane & 1) ? (xr * s + xi * c) : (xr * c - xi * s);
        value = roundtrip_scalar(value, q_dtype);
    }

    values[lane] = value;
    __syncthreads();

#pragma unroll
    for (int span = 1; span < kHeadDim; span <<= 1) {
        const int group = lane / (span << 1);
        const int offset = lane & ((span << 1) - 1);
        if (offset < span) {
            const int a_idx = group * (span << 1) + offset;
            const int b_idx = a_idx + span;
            const float a = values[a_idx];
            const float b = values[b_idx];
            values[a_idx] = a + b;
            values[b_idx] = a - b;
        }
        __syncthreads();
    }

    value = roundtrip_scalar(values[lane] * kHadamardScale, q_dtype);
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

void launch_fused_q_indexer_rope_hadamard_quant(const void *q,
                                                int q_dtype,
                                                const void *weights,
                                                int weights_dtype,
                                                uint8_t *q_fp8,
                                                float *q_scale,
                                                float *fused_weights,
                                                float weight_scale,
                                                const float *freqs_cis,
                                                const void *positions,
                                                bool positions_i64,
                                                int64_t rows,
                                                int64_t heads,
                                                void *stream) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    fused_q_indexer_rope_hadamard_quant_kernel<<<static_cast<unsigned int>(rows), kHeadDim, 0, cuda_stream>>>(
        q,
        q_dtype,
        weights,
        weights_dtype,
        q_fp8,
        q_scale,
        fused_weights,
        weight_scale,
        freqs_cis,
        positions,
        positions_i64,
        rows,
        heads);
    return;
}

} // namespace infinicore::op::deepseek_v4_fused_q_indexer_rope_hadamard_quant
