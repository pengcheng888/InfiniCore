#include "deepseek_v4_embedding_and_hc_expand_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace infinicore::op::deepseek_v4_embedding_and_hc_expand_kernel_impl {
namespace {

constexpr int kThreads = 256;
constexpr int kVectorBytes = 16;
constexpr int kDsv4Hidden = 4096;
constexpr int kDsv4HcMult = 4;
constexpr int kDsv4Bf16VectorsPerRow = kDsv4Hidden * static_cast<int>(sizeof(__nv_bfloat16)) / kVectorBytes;

template <typename T, typename IndexT>
__global__ void embedding_scalar_kernel(T *__restrict__ out,
                                        const IndexT *__restrict__ input,
                                        const T *__restrict__ weight,
                                        int64_t tokens,
                                        int64_t hc_mult,
                                        int64_t hidden,
                                        int64_t vocab) {
    const int64_t token = static_cast<int64_t>(blockIdx.x);
    if (token >= tokens) {
        return;
    }
    const int64_t index = static_cast<int64_t>(input[token]);
    if (index < 0 || index >= vocab) {
        return;
    }
    const int64_t weight_base = index * hidden;
    for (int64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
        const T value = weight[weight_base + h];
        for (int64_t hc = 0; hc < hc_mult; ++hc) {
            out[(token * hc_mult + hc) * hidden + h] = value;
        }
    }
}

template <typename T, typename IndexT>
__global__ void embedding_vector_kernel(T *__restrict__ out,
                                        const IndexT *__restrict__ input,
                                        const T *__restrict__ weight,
                                        int64_t tokens,
                                        int64_t hc_mult,
                                        int64_t hidden,
                                        int64_t vocab,
                                        int64_t vectors_per_row) {
    const int64_t token = static_cast<int64_t>(blockIdx.x);
    if (token >= tokens) {
        return;
    }
    const int64_t index = static_cast<int64_t>(input[token]);
    if (index < 0 || index >= vocab) {
        return;
    }
    auto *out_vec = reinterpret_cast<uint4 *>(out + token * hc_mult * hidden);
    const auto *weight_vec = reinterpret_cast<const uint4 *>(weight + index * hidden);
    for (int64_t vec = threadIdx.x; vec < vectors_per_row; vec += blockDim.x) {
        const uint4 value = weight_vec[vec];
        for (int64_t hc = 0; hc < hc_mult; ++hc) {
            out_vec[hc * vectors_per_row + vec] = value;
        }
    }
}

template <typename IndexT>
__global__ void embedding_dsv4_bf16_h4096_hc4_kernel(__nv_bfloat16 *__restrict__ out,
                                                     const IndexT *__restrict__ input,
                                                     const __nv_bfloat16 *__restrict__ weight,
                                                     int64_t vocab) {
    const int64_t token = static_cast<int64_t>(blockIdx.x);
    const int64_t index = static_cast<int64_t>(input[token]);
    if (index < 0 || index >= vocab) {
        return;
    }

    auto *out_vec = reinterpret_cast<uint4 *>(out + token * kDsv4HcMult * kDsv4Hidden);
    const auto *weight_vec = reinterpret_cast<const uint4 *>(weight + index * kDsv4Hidden);
    for (int vec = threadIdx.x; vec < kDsv4Bf16VectorsPerRow; vec += blockDim.x) {
        const uint4 value = weight_vec[vec];
        out_vec[vec] = value;
        out_vec[kDsv4Bf16VectorsPerRow + vec] = value;
        out_vec[2 * kDsv4Bf16VectorsPerRow + vec] = value;
        out_vec[3 * kDsv4Bf16VectorsPerRow + vec] = value;
    }
}

template <typename T, typename IndexT>
void launch_typed(void *out,
                  const void *input,
                  const void *weight,
                  int64_t tokens,
                  int64_t hc_mult,
                  int64_t hidden,
                  int64_t vocab,
                  cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned int>(tokens));
    const dim3 block(kThreads);
    constexpr int64_t elements_per_vector = kVectorBytes / static_cast<int64_t>(sizeof(T));
    if (hidden % elements_per_vector == 0) {
        embedding_vector_kernel<T, IndexT><<<grid, block, 0, stream>>>(
            reinterpret_cast<T *>(out),
            reinterpret_cast<const IndexT *>(input),
            reinterpret_cast<const T *>(weight),
            tokens,
            hc_mult,
            hidden,
            vocab,
            hidden / elements_per_vector);
        return;
    }
    embedding_scalar_kernel<T, IndexT><<<grid, block, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const IndexT *>(input),
        reinterpret_cast<const T *>(weight),
        tokens,
        hc_mult,
        hidden,
        vocab);
}

template <typename T>
void launch_for_index(void *out,
                      const void *input,
                      const void *weight,
                      int64_t tokens,
                      int64_t hc_mult,
                      int64_t hidden,
                      int64_t vocab,
                      DataType input_dtype,
                      cudaStream_t stream) {
    if (input_dtype == DataType::I32) {
        launch_typed<T, int32_t>(out, input, weight, tokens, hc_mult, hidden, vocab, stream);
        return;
    }
    if (input_dtype == DataType::I64) {
        launch_typed<T, int64_t>(out, input, weight, tokens, hc_mult, hidden, vocab, stream);
        return;
    }
    throw std::runtime_error("deepseek_v4_embedding_and_hc_expand_kernel_ expects int32 or int64 input indices.");
}

template <typename IndexT>
void launch_dsv4_bf16_h4096_hc4(void *out,
                                const void *input,
                                const void *weight,
                                int64_t tokens,
                                int64_t vocab,
                                cudaStream_t stream) {
    embedding_dsv4_bf16_h4096_hc4_kernel<IndexT><<<
        dim3(static_cast<unsigned int>(tokens)),
        dim3(kThreads),
        0,
        stream>>>(
        reinterpret_cast<__nv_bfloat16 *>(out),
        reinterpret_cast<const IndexT *>(input),
        reinterpret_cast<const __nv_bfloat16 *>(weight),
        vocab);
}

bool try_launch_dsv4_fixed_shape(void *out,
                                 const void *input,
                                 const void *weight,
                                 int64_t tokens,
                                 int64_t hc_mult,
                                 int64_t hidden,
                                 int64_t vocab,
                                 DataType out_dtype,
                                 DataType input_dtype,
                                 cudaStream_t stream) {
    if (out_dtype != DataType::BF16 || hidden != kDsv4Hidden || hc_mult != kDsv4HcMult) {
        return false;
    }
    if (input_dtype == DataType::I64) {
        launch_dsv4_bf16_h4096_hc4<int64_t>(out, input, weight, tokens, vocab, stream);
        return true;
    }
    if (input_dtype == DataType::I32) {
        launch_dsv4_bf16_h4096_hc4<int32_t>(out, input, weight, tokens, vocab, stream);
        return true;
    }
    return false;
}

} // namespace

void launch_embedding(void *out,
                      const void *input,
                      const void *weight,
                      int64_t tokens,
                      int64_t hc_mult,
                      int64_t hidden,
                      int64_t vocab,
                      DataType out_dtype,
                      DataType input_dtype,
                      void *stream) {
    if (tokens <= 0 || hc_mult <= 0 || hidden <= 0 || vocab <= 0) {
        return;
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (try_launch_dsv4_fixed_shape(
            out,
            input,
            weight,
            tokens,
            hc_mult,
            hidden,
            vocab,
            out_dtype,
            input_dtype,
            cuda_stream)) {
        return;
    }
    if (out_dtype == DataType::BF16) {
        launch_for_index<__nv_bfloat16>(out, input, weight, tokens, hc_mult, hidden, vocab, input_dtype, cuda_stream);
        return;
    }
    if (out_dtype == DataType::F16) {
        launch_for_index<__half>(out, input, weight, tokens, hc_mult, hidden, vocab, input_dtype, cuda_stream);
        return;
    }
    if (out_dtype == DataType::F32) {
        launch_for_index<float>(out, input, weight, tokens, hc_mult, hidden, vocab, input_dtype, cuda_stream);
        return;
    }
    throw std::runtime_error("deepseek_v4_embedding_and_hc_expand_kernel_ supports bf16/fp16/fp32 output tensors only.");
}

} // namespace infinicore::op::deepseek_v4_embedding_and_hc_expand_kernel_impl
