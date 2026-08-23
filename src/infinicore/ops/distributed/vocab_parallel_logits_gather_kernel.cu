#include "vocab_parallel_logits_gather_kernel.hpp"

#include <cuda_runtime.h>
#include <stdint.h>

namespace infinicore::op::distributed {
namespace {

constexpr int kThreads = 256;

__global__ void reorder_bytes_kernel(uint8_t *__restrict__ output,
                                     const uint8_t *__restrict__ gathered,
                                     size_t total_elements,
                                     size_t tokens,
                                     size_t vocab_local,
                                     size_t world_size,
                                     size_t element_size) {
    const size_t stride = static_cast<size_t>(blockDim.x) * blockIdx.x;
    const size_t step = static_cast<size_t>(blockDim.x) * gridDim.x;
    const size_t vocab_global = vocab_local * world_size;

    for (size_t idx = stride + threadIdx.x; idx < total_elements; idx += step) {
        const size_t col = idx % vocab_local;
        const size_t row = idx / vocab_local;
        const size_t token = row % tokens;
        const size_t rank = row / tokens;
        const size_t dst = token * vocab_global + rank * vocab_local + col;

        const uint8_t *src_ptr = gathered + idx * element_size;
        uint8_t *dst_ptr = output + dst * element_size;
        for (size_t byte = 0; byte < element_size; ++byte) {
            dst_ptr[byte] = src_ptr[byte];
        }
    }
}

} // namespace

void launch_vocab_parallel_logits_reorder(void *output,
                                          const void *gathered,
                                          size_t tokens,
                                          size_t vocab_local,
                                          size_t world_size,
                                          size_t element_size,
                                          void *stream) {
    if (tokens == 0 || vocab_local == 0 || world_size <= 1 || element_size == 0) {
        return;
    }

    const size_t total_elements = tokens * vocab_local * world_size;
    const size_t blocks = (total_elements + kThreads - 1) / kThreads;
    const unsigned int grid = static_cast<unsigned int>(blocks > 65535 ? 65535 : blocks);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    reorder_bytes_kernel<<<grid, kThreads, 0, cuda_stream>>>(
        reinterpret_cast<uint8_t *>(output),
        reinterpret_cast<const uint8_t *>(gathered),
        total_elements,
        tokens,
        vocab_local,
        world_size,
        element_size);
}

} // namespace infinicore::op::distributed
