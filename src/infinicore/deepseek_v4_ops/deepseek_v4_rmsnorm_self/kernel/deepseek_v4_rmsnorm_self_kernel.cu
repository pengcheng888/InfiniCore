#include "deepseek_v4_rmsnorm_self_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace infinicore::op::deepseek_v4_rmsnorm_self_native {
namespace {

__device__ __forceinline__ float load_scalar(const void *__restrict__ ptr, int64_t idx, int dtype) {
    if (dtype == kDsv4BF16) {
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(ptr)[idx]);
    }
    if (dtype == kDsv4F16) {
        return __half2float(reinterpret_cast<const __half *>(ptr)[idx]);
    }
    return reinterpret_cast<const float *>(ptr)[idx];
}

__device__ __forceinline__ void store_scalar(void *__restrict__ ptr, int64_t idx, int dtype, float value) {
    if (dtype == kDsv4BF16) {
        reinterpret_cast<__nv_bfloat16 *>(ptr)[idx] = __float2bfloat16(value);
    } else if (dtype == kDsv4F16) {
        reinterpret_cast<__half *>(ptr)[idx] = __float2half(value);
    } else {
        reinterpret_cast<float *>(ptr)[idx] = value;
    }
}

__global__ void rmsnorm_self_kernel(void *__restrict__ out,
                                    const void *__restrict__ input,
                                    int dtype,
                                    int64_t rows,
                                    int64_t dim,
                                    float epsilon) {
    const int64_t row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= rows) {
        return;
    }

    extern __shared__ float smem[];
    float local_sum = 0.0f;
    const int64_t base = row * dim;
    for (int64_t i = lane; i < dim; i += blockDim.x) {
        const float v = load_scalar(input, base + i, dtype);
        local_sum += v * v;
    }
    smem[lane] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            smem[lane] += smem[lane + stride];
        }
        __syncthreads();
    }

    const float inv = rsqrtf(smem[0] / static_cast<float>(dim) + epsilon);
    for (int64_t i = lane; i < dim; i += blockDim.x) {
        const float v = load_scalar(input, base + i, dtype) * inv;
        store_scalar(out, base + i, dtype, v);
    }
}

} // namespace

void launch_rmsnorm_self(void *out,
                         const void *input,
                         int dtype,
                         int64_t rows,
                         int64_t dim,
                         float epsilon,
                         void *stream) {
    if (rows <= 0 || dim <= 0) {
        return;
    }
    constexpr int threads = 256;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    rmsnorm_self_kernel<<<static_cast<unsigned int>(rows), threads, threads * sizeof(float), cuda_stream>>>(
        out, input, dtype, rows, dim, epsilon);
}

} // namespace infinicore::op::deepseek_v4_rmsnorm_self_native
