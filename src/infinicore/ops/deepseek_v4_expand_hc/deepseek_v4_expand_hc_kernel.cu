#include "deepseek_v4_expand_hc_kernel.hpp"

#include <cuda_runtime.h>
#include <stdint.h>

namespace infinicore::op::deepseek_v4_expand_hc_impl {
namespace {

constexpr int kBlockSize = 256;

template <typename T>
__global__ void expand_hc_kernel(T *__restrict__ output,
                                 const T *__restrict__ input,
                                 int64_t total,
                                 int64_t hc,
                                 int64_t hidden) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int64_t col = idx % hidden;
    const int64_t token = idx / (hc * hidden);
    output[idx] = input[token * hidden + col];
}

} // namespace

void launch_expand_hc(void *output,
                      const void *input,
                      int64_t tokens,
                      int64_t hc,
                      int64_t hidden,
                      int element_size,
                      void *stream) {
    const int64_t total = tokens * hc * hidden;
    const int blocks = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
    if (element_size == 2) {
        expand_hc_kernel<uint16_t><<<blocks, kBlockSize, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<uint16_t *>(output),
            reinterpret_cast<const uint16_t *>(input),
            total,
            hc,
            hidden);
    } else if (element_size == 4) {
        expand_hc_kernel<uint32_t><<<blocks, kBlockSize, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<uint32_t *>(output),
            reinterpret_cast<const uint32_t *>(input),
            total,
            hc,
            hidden);
    }
}

} // namespace infinicore::op::deepseek_v4_expand_hc_impl
