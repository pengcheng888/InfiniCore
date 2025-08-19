#ifndef _TOPKSOFTMAX_KERNEL_CUH__
#define _TOPKSOFTMAX_KERNEL_CUH__
#include <cuda_runtime.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_store.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

inline __device__ float exp_func(float x) {
    return __expf(x);
}

inline __device__ __nv_bfloat16 exp_func(__nv_bfloat16 x) {
    return __float2bfloat16(__expf(__bfloat162float(x)));
}

inline __device__ half exp_func(half x) {
    return hexp(x);
}

//
// 对每一行数据 softmax
//
template <typename T, int BLOCK_SIZE = 128>
__global__ void softmax_row_kernel(T *output,      // 输出数据 [N, width]
                                   T *input,       // 输入数据 [N, width]
                                   const int N,    // 总行数
                                   const int width // 每行元素数量

) {
    const int bid = blockIdx.x; // 当前行索引
    if (bid >= N) {
        return;
    }

    const int tid = threadIdx.x; // 当前元素在行内的位置
    T *data_input = input + bid * width;
    T *data_output = output + bid * width;

    // 声明共享内存存储中间结果
    __shared__ T shared_max;
    __shared__ T shared_sum;
    typedef cub::BlockReduce<T, BLOCK_SIZE> BlockReduce;

    // ------------------------------------------------ //
    //             第一步：计算最大值                      //
    // ------------------------------------------------ //
    constexpr float kHalfMinNorm = 6.10351562e-05F;     // FP16 最小规格化数
    constexpr float kBfloat16MinNorm = 1.17549435e-38F; // BF16 最小规格化数
    T thread_max;

    if constexpr (std::is_same_v<T, float>) {
        thread_max = -FLT_MAX;
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        thread_max = __float2bfloat16(kBfloat16MinNorm);
    } else if constexpr (std::is_same_v<T, half>) {
        thread_max = __float2half(kHalfMinNorm);
    }

    for (int i = tid; i < width; i += BLOCK_SIZE) {
        thread_max = thread_max > data_input[i] ? thread_max : data_input[i];
    }

    {
        __shared__ typename BlockReduce::TempStorage temp_storage_max;
        T value_max = BlockReduce(temp_storage_max).Reduce(thread_max, cub::Max());
        if (tid == 0) {
            shared_max = value_max;
        }
    }
    __syncthreads();

    // ------------------------------------------------ //
    //             第二步：计算指数和                      //
    // ------------------------------------------------ //
    T exp_val;
    if constexpr (std::is_same_v<T, float>) {
        exp_val = 0.0f;
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        exp_val = __float2bfloat16(0.0f);
    } else if constexpr (std::is_same_v<T, half>) {
        exp_val = __float2half(0.0f);
    }

    for (int i = tid; i < width; i += BLOCK_SIZE) {
        T temp_val = data_input[i] - shared_max;
        exp_val += exp_func(temp_val);
    }

    {
        __shared__ typename BlockReduce::TempStorage temp_storage_sum;
        T value_sum = BlockReduce(temp_storage_sum).Sum(exp_val);
        if (tid == 0) {
            shared_sum = value_sum;
        }
    }
    __syncthreads();

    // ------------------------------------------------ //
    //           第三步：计算 Softmax                     //
    // ------------------------------------------------ //
    for (int i = tid; i < width; i += BLOCK_SIZE) {
        T temp_val = data_input[i] - shared_max;
        data_output[i] = exp_func(temp_val) / shared_sum;
    }
}

template <typename T, int BLOCK_SIZE = 128>
__global__ void topk_row_kernel(T *values_topk,    // 输出值, 形状[N, TOPK]
                                int *indices_topk, // 输出索引, 形状[N, TOPK]
                                T *input,          // 输入数据, 形状[N, width]
                                const int N,       // 行数 (N)
                                const int width,   // 每行的元素数量
                                const int topk,
                                const bool norm_topk = false) {
    const int bid = blockIdx.x;
    if (bid >= N) {
        return;
    }

    int tid = threadIdx.x;
    T *data_input = input + bid * width;
    T *values_topk_output = values_topk + bid * topk;
    int *indices_topk_output = indices_topk + bid * topk;

    ///////////////////////////////////////////////////////
    constexpr float kHalfMinNorm = 6.10351562e-05F;     // FP16 最小规格化数
    constexpr float kBfloat16MinNorm = 1.17549435e-38F; // BF16 最小规格化数

    int thread_indices[1]; // 每个线程处理 1个值
    T thread_values[1];    // 每个线程处理 1个键
    thread_indices[0] = -1;
    if constexpr (std::is_same_v<T, float>) {
        thread_values[0] = -FLT_MAX;
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        thread_values[0] = __float2bfloat16(kBfloat16MinNorm);
    } else if constexpr (std::is_same_v<T, half>) {
        thread_values[0] = __float2half(kHalfMinNorm);
    }

    // ------------------------------------------------- //
    //             1. 加载数据到线程寄存器                   //
    // ------------------------------------------------- //
    T temp;
    for (int i = tid; i < width; i += BLOCK_SIZE) { // 大于128怎么办法？未解决
        temp = data_input[i];
        if (temp > thread_values[0]) {
            thread_values[0] = temp;
            thread_indices[0] = i;
        }
    }

    // ------------------------------------------------- //
    //             2. 使用CUB块级排序 (降序)                //
    // ------------------------------------------------- //
    typedef cub::BlockRadixSort<T, BLOCK_SIZE, 1, int> BlockRadixSort;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;
    BlockRadixSort(temp_storage).SortDescending(thread_values, thread_indices);
    __syncthreads();

    // ------------------------------------------------- //
    //              3. 做归一化 (warp粒度??)               //
    // ------------------------------------------------- //
    T val = 0.0;
    for (int i = tid; i < width; i += BLOCK_SIZE) {
        val += thread_values[0];
        //  break;
    }
    __shared__ T shared_norm;
    {
        // 改称 warp中的排序，还未改动
        typedef cub::BlockReduce<T, BLOCK_SIZE> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage_sum;
        T value_sum = BlockReduce(temp_storage_sum).Sum(val);
        if (tid == 0) {
            shared_norm = value_sum;
        }
    }
    __syncthreads();

    // ------------------------------------------------- //
    //             4. 前K个线程写入TopK结果                 //
    // ------------------------------------------------- //
    if (false == norm_topk) {
        shared_norm = 1.0;
    }

    if (tid < topk) {
        values_topk_output[tid] = thread_values[0]; //  * shared_norm
        indices_topk_output[tid] = thread_indices[0];
    }
}

#endif // _TOPKSOFTMAX_KERNEL_CUH__
