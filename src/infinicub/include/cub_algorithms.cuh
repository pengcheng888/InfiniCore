#ifndef __CUB_ALGORITHMS_CUH__
#define __CUB_ALGORITHMS_CUH__

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cub/cub.cuh>

namespace infini_cub {

cudaError cub_DeviceReduce_ArgMax(
    cub::KeyValuePair<int, __nv_bfloat16>* kv_pair,
    const __nv_bfloat16* logits,
    int n,
    void* workspace_ptr,
    size_t& workspace_len,
    cudaStream_t stream);

cudaError cub_DeviceReduce_ArgMax(
    cub::KeyValuePair<int, half>* kv_pair,
    const half* logits,
    int n,
    void* workspace_ptr,
    size_t& workspace_len,
    cudaStream_t stream);

cudaError cub_DeviceReduce_ArgMax(
    cub::KeyValuePair<int, float>* kv_pair,
    const float* logits,
    int n,
    void* workspace_ptr,
    size_t& workspace_len,
    cudaStream_t stream);

cudaError cub_DeviceReduce_ArgMax(
    cub::KeyValuePair<int, double>* kv_pair,
    const double* logits,
    int n,
    void* workspace_ptr,
    size_t& workspace_len,
    cudaStream_t stream);
}  // namespace infini_cub

namespace infini_cub {

cudaError cub_DeviceScan_InclusiveSum(
    void* workspace_ptr,
    size_t& workspace_len,
    __nv_bfloat16* data,
    int n,
    cudaStream_t stream);

cudaError cub_DeviceScan_InclusiveSum(
    void* workspace_ptr,
    size_t& workspace_len,
    half* data,
    int n,
    cudaStream_t stream);

cudaError cub_DeviceScan_InclusiveSum(
    void* workspace_ptr,
    size_t& workspace_len,
    float* data,
    int n,
    cudaStream_t stream);

cudaError cub_DeviceScan_InclusiveSum(
    void* workspace_ptr,
    size_t& workspace_len,
    double* data,
    int n,
    cudaStream_t stream);
}  // namespace infini_cub

namespace infini_cub {
using uchar = uint8_t;
using ushort = uint16_t;
using uint = uint32_t;
using ulong = uint64_t;

cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const __nv_bfloat16* key_in,
    __nv_bfloat16* key_out,
    const uchar* val_in,
    uchar* val_out,
    int n,
    cudaStream_t stream);

cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const __nv_bfloat16* key_in,
    __nv_bfloat16* key_out,
    const int8_t* val_in,
    int8_t* val_out,
    int n,
    cudaStream_t stream);

cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const __nv_bfloat16* key_in,
    __nv_bfloat16* key_out,
    const ushort* val_in,
    ushort* val_out,
    int n,
    cudaStream_t stream);

cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const __nv_bfloat16* key_in,
    __nv_bfloat16* key_out,
    const short* val_in,
    short* val_out,
    int n,
    cudaStream_t stream);

cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const __nv_bfloat16* key_in,
    __nv_bfloat16* key_out,
    const uint* val_in,
    uint* val_out,
    int n,
    cudaStream_t stream);

cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const __nv_bfloat16* key_in,
    __nv_bfloat16* key_out,
    const int* val_in,
    int* val_out,
    int n,
    cudaStream_t stream);
cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const __nv_bfloat16* key_in,
    __nv_bfloat16* key_out,
    const ulong* val_in,
    ulong* val_out,
    int n,
    cudaStream_t stream);

cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const __nv_bfloat16* key_in,
    __nv_bfloat16* key_out,
    const long* val_in,
    long* val_out,
    int n,
    cudaStream_t stream);
// --------------------------------------------------------------
// --------------------------------------------------------------
// --------------------------------------------------------------
cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const half* key_in,
    half* key_out,
    const uchar* val_in,
    uchar* val_out,
    int n,
    cudaStream_t stream);

cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const half* key_in,
    half* key_out,
    const int8_t* val_in,
    int8_t* val_out,
    int n,
    cudaStream_t stream);

cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const half* key_in,
    half* key_out,
    const ushort* val_in,
    ushort* val_out,
    int n,
    cudaStream_t stream);
cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const half* key_in,
    half* key_out,
    const short* val_in,
    short* val_out,
    int n,
    cudaStream_t stream);

cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const half* key_in,
    half* key_out,
    const uint* val_in,
    uint* val_out,
    int n,
    cudaStream_t stream);

cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const half* key_in,
    half* key_out,
    const int* val_in,
    int* val_out,
    int n,
    cudaStream_t stream);

cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const half* key_in,
    half* key_out,
    const ulong* val_in,
    ulong* val_out,
    int n,
    cudaStream_t stream);
cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const half* key_in,
    half* key_out,
    const long* val_in,
    long* val_out,
    int n,
    cudaStream_t stream);
// --------------------------------------------------------------
// --------------------------------------------------------------
// --------------------------------------------------------------
cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const float* key_in,
    float* key_out,
    const uchar* val_in,
    uchar* val_out,
    int n,
    cudaStream_t stream);

cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const float* key_in,
    float* key_out,
    const int8_t* val_in,
    int8_t* val_out,
    int n,
    cudaStream_t stream);

cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const float* key_in,
    float* key_out,
    const ushort* val_in,
    ushort* val_out,
    int n,
    cudaStream_t stream);

cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const float* key_in,
    float* key_out,
    const short* val_in,
    short* val_out,
    int n,
    cudaStream_t stream);
cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const float* key_in,
    float* key_out,
    const uint* val_in,
    uint* val_out,
    int n,
    cudaStream_t stream);

cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const float* key_in,
    float* key_out,
    const int* val_in,
    int* val_out,
    int n,
    cudaStream_t stream);

cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const float* key_in,
    float* key_out,
    const ulong* val_in,
    ulong* val_out,
    int n,
    cudaStream_t stream);
cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const float* key_in,
    float* key_out,
    const long* val_in,
    long* val_out,
    int n,
    cudaStream_t stream);
// --------------------------------------------------------------
// --------------------------------------------------------------
// --------------------------------------------------------------
cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const double* key_in,
    double* key_out,
    const uchar* val_in,
    uchar* val_out,
    int n,
    cudaStream_t stream);

cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const double* key_in,
    double* key_out,
    const int8_t* val_in,
    int8_t* val_out,
    int n,
    cudaStream_t stream);
cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const double* key_in,
    double* key_out,
    const ushort* val_in,
    ushort* val_out,
    int n,
    cudaStream_t stream);

cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const double* key_in,
    double* key_out,
    const short* val_in,
    short* val_out,
    int n,
    cudaStream_t stream);

cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const double* key_in,
    double* key_out,
    const uint* val_in,
    uint* val_out,
    int n,
    cudaStream_t stream);

cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const double* key_in,
    double* key_out,
    const int* val_in,
    int* val_out,
    int n,
    cudaStream_t stream);

cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const double* key_in,
    double* key_out,
    const ulong* val_in,
    ulong* val_out,
    int n,
    cudaStream_t stream);
cudaError cub_DeviceRadixSort_SortPairsDescending(
    void* workspace_ptr,
    size_t& workspace_len,
    const double* key_in,
    double* key_out,
    const long* val_in,
    long* val_out,
    int n,
    cudaStream_t stream);

}  // namespace infini_cub

#endif  // __CUB_ALGORITHMS_CUH__