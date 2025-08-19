#include "../../../devices/nvidia/nvidia_common.cuh"
#include "topksoftmax_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cub/block/block_reduce.cuh>

#include "../../../reduce/cuda/reduce.cuh"

#include "../cuda/kernel.cuh"

namespace op::topksoftmax::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t indices_desc,
    infiniopTensorDescriptor_t x_desc) {
    auto result = TopksoftmaxInfo::create(values_desc, indices_desc, x_desc);
    CHECK_RESULT(result);
    auto info = result.take();

    // only support contiguous last dimension
    if (info.x_strides[1] != 1 || info.y_strides[1] != 1) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        std::move(info),
        0,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

namespace {
// void launch_softmax_row(float *d_output, float *d_input, const int N, const int width) {
//     const int block_threads = 128;

//     // 每个线程块处理一行
//     dim3 blocks(N);
//     dim3 threads(block_threads);

//     // 根据块大小分派内核
//     softmax_row_kernel<float><<<blocks, threads>>>(d_output, d_input, N, width);
// }

// 封装函数
// template <typename T>
// void launch_topk_row(T *d_values_out, int *d_indices_out, T *d_input, int N, int width, int topk) {

//     const int block_threads = 128;

//     // 每个线程块处理一行
//     dim3 blocks(N);
//     dim3 threads(block_threads);

//     topk_row_kernel<T><<<blocks, threads>>>(d_values_out, d_indices_out, d_input, N, width, topk, false);
//     cudaDeviceSynchronize();
// }
infiniStatus_t launch_topksoftmax(void *d_values_out, void *d_indices_out, void *d_input, int N, int width, int topk, infiniDtype_t xtype,
                                  cudaStream_t stream) {

    const int block_threads = 128;
    dim3 blocks(N);
    dim3 threads(block_threads);

    if (xtype == INFINI_DTYPE_F32) {
        printf(" xtype == INFINI_DTYPE_F32 \n");
        softmax_row_kernel<float><<<blocks, threads, 0, stream>>>((float *)d_input, (float *)d_input, N, width);
        topk_row_kernel<float><<<blocks, threads, 0, stream>>>((float *)d_values_out, (int *)d_indices_out, (float *)d_input, N, width, topk, false);
    } else if (xtype == INFINI_DTYPE_F16) {
        printf(" xtype == INFINI_DTYPE_F16 \n");
        softmax_row_kernel<half><<<blocks, threads, 0, stream>>>((half *)d_input, (half *)d_input, N, width);
        topk_row_kernel<half><<<blocks, threads, 0, stream>>>((half *)d_values_out, (int *)d_indices_out, (half *)d_input, N, width, topk, false);
    } else if (xtype == INFINI_DTYPE_BF16) {
        printf(" xtype == INFINI_DTYPE_BF16 \n");
        softmax_row_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>((__nv_bfloat16 *)d_input, (__nv_bfloat16 *)d_input, N, width);
        topk_row_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>((__nv_bfloat16 *)d_values_out, (int *)d_indices_out, (__nv_bfloat16 *)d_input, N, width, topk, false);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}
}; // namespace

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *values, void *indices, void *x,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    int N = _info.N;
    int width = _info.width;
    int topk = _info.topk;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // auto stride_x = _info.x_strides[0];
    // auto stride_y = _info.y_strides[0];
    // auto dim = _info.dim();
    // uint32_t batch_size = static_cast<uint32_t>(_info.shape[0]);

    // // launch kernel with different block sizes
    // if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
    //     CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_1024>(batch_size, dim, y, _info.atype, stride_y, x, stride_x, w, _info.wtype, _info.epsilon, cuda_stream));
    // } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
    //     CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_512>(batch_size, dim, y, _info.atype, stride_y, x, stride_x, w, _info.wtype, _info.epsilon, cuda_stream));
    // } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
    //     CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_4096>(batch_size, dim, y, _info.atype, stride_y, x, stride_x, w, _info.wtype, _info.epsilon, cuda_stream));
    // } else {
    //     return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    // }

    return launch_topksoftmax(values, indices, x, N, width, topk, _info.xtype, cuda_stream);
}
} // namespace op::topksoftmax::nvidia
