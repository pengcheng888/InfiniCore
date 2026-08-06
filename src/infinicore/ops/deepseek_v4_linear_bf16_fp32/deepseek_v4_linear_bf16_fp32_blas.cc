#include "infinicore/ops/deepseek_v4_linear_bf16_fp32.hpp"

#include "deepseek_v4_linear_bf16_fp32_common.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#if defined(ENABLE_HYGON_API)
#include <hipblas/hipblas.h>
#elif defined(ENABLE_NVIDIA_API)
#include <cublas_v2.h>
#endif

#include <stdexcept>
#include <string>

namespace infinicore::op {
namespace {

void check_accelerator_tensor(const Tensor &tensor, const char *op_name) {
#if defined(ENABLE_HYGON_API)
    if (tensor->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error(std::string(op_name) + " expects HYGON tensors in this build.");
    }
#elif defined(ENABLE_NVIDIA_API)
    if (tensor->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error(std::string(op_name) + " expects NVIDIA tensors in this build.");
    }
#else
    (void)tensor;
    (void)op_name;
#endif
}

void check_blas_tensors(const Tensor &out, const Tensor &x, const Tensor &weight, const char *op_name) {
    check_accelerator_tensor(x, op_name);
    deepseek_v4_linear_bf16_fp32_impl::check_shapes(out, x, weight, op_name);
    if (x->dtype() != DataType::BF16 || weight->dtype() != DataType::BF16) {
        throw std::runtime_error(std::string(op_name) + " expects bf16 input and weight tensors.");
    }
    if (!out->is_contiguous() || !x->is_contiguous() || !weight->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous tensors.");
    }
}

#if defined(ENABLE_HYGON_API)

const char *blas_status_name(hipblasStatus_t status) {
    switch (status) {
    case HIPBLAS_STATUS_SUCCESS:
        return "HIPBLAS_STATUS_SUCCESS";
    case HIPBLAS_STATUS_NOT_INITIALIZED:
        return "HIPBLAS_STATUS_NOT_INITIALIZED";
    case HIPBLAS_STATUS_ALLOC_FAILED:
        return "HIPBLAS_STATUS_ALLOC_FAILED";
    case HIPBLAS_STATUS_INVALID_VALUE:
        return "HIPBLAS_STATUS_INVALID_VALUE";
    case HIPBLAS_STATUS_MAPPING_ERROR:
        return "HIPBLAS_STATUS_MAPPING_ERROR";
    case HIPBLAS_STATUS_EXECUTION_FAILED:
        return "HIPBLAS_STATUS_EXECUTION_FAILED";
    case HIPBLAS_STATUS_INTERNAL_ERROR:
        return "HIPBLAS_STATUS_INTERNAL_ERROR";
    case HIPBLAS_STATUS_NOT_SUPPORTED:
        return "HIPBLAS_STATUS_NOT_SUPPORTED";
    default:
        return "HIPBLAS_STATUS_UNKNOWN";
    }
}

hipblasHandle_t get_blas_handle() {
    thread_local hipblasHandle_t handle = [] {
        hipblasHandle_t created = nullptr;
        auto status = hipblasCreate(&created);
        if (status != HIPBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("hipblasCreate failed: ") + blas_status_name(status));
        }
        return created;
    }();
    return handle;
}

void launch_blas_linear_bf16_fp32(float *out,
                                  const void *x,
                                  const void *weight,
                                  int64_t tokens,
                                  int64_t out_features,
                                  int64_t in_features) {
    auto handle = get_blas_handle();
    auto stream = reinterpret_cast<hipStream_t>(context::getStream());
    auto status = hipblasSetStream(handle, stream);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("hipblasSetStream failed: ") + blas_status_name(status));
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    status = hipblasGemmEx(handle,
                           HIPBLAS_OP_T,
                           HIPBLAS_OP_N,
                           static_cast<int>(out_features),
                           static_cast<int>(tokens),
                           static_cast<int>(in_features),
                           &alpha,
                           weight,
                           HIPBLAS_R_16B,
                           static_cast<int>(in_features),
                           x,
                           HIPBLAS_R_16B,
                           static_cast<int>(in_features),
                           &beta,
                           out,
                           HIPBLAS_R_32F,
                           static_cast<int>(out_features),
                           HIPBLAS_R_32F,
                           HIPBLAS_GEMM_DEFAULT);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("hipblasGemmEx failed: ") + blas_status_name(status));
    }
}

#elif defined(ENABLE_NVIDIA_API)

const char *blas_status_name(cublasStatus_t status) {
    switch (status) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    default:
        return "CUBLAS_STATUS_UNKNOWN";
    }
}

cublasHandle_t get_blas_handle() {
    thread_local cublasHandle_t handle = [] {
        cublasHandle_t created = nullptr;
        auto status = cublasCreate(&created);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("cublasCreate failed: ") + blas_status_name(status));
        }
        return created;
    }();
    return handle;
}

void launch_blas_linear_bf16_fp32(float *out,
                                  const void *x,
                                  const void *weight,
                                  int64_t tokens,
                                  int64_t out_features,
                                  int64_t in_features) {
    auto handle = get_blas_handle();
    auto stream = reinterpret_cast<cudaStream_t>(context::getStream());
    auto status = cublasSetStream(handle, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cublasSetStream failed: ") + blas_status_name(status));
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    status = cublasGemmEx(handle,
                          CUBLAS_OP_T,
                          CUBLAS_OP_N,
                          static_cast<int>(out_features),
                          static_cast<int>(tokens),
                          static_cast<int>(in_features),
                          &alpha,
                          weight,
                          CUDA_R_16BF,
                          static_cast<int>(in_features),
                          x,
                          CUDA_R_16BF,
                          static_cast<int>(in_features),
                          &beta,
                          out,
                          CUDA_R_32F,
                          static_cast<int>(out_features),
                          CUBLAS_COMPUTE_32F,
                          CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cublasGemmEx failed: ") + blas_status_name(status));
    }
}

#endif

} // namespace

Tensor deepseek_v4_linear_bf16_fp32_blas(const Tensor &x, const Tensor &weight) {
    if (x->ndim() != 2 || weight->ndim() != 2) {
        throw std::runtime_error("deepseek_v4_linear_bf16_fp32_blas expects 2D input and weight tensors.");
    }
    if (x->size(1) != weight->size(1)) {
        throw std::runtime_error("deepseek_v4_linear_bf16_fp32_blas input/weight K dimension mismatch.");
    }
    auto out = Tensor::empty({x->size(0), weight->size(0)}, DataType::F32, x->device());
    deepseek_v4_linear_bf16_fp32_blas_(out, x, weight);
    return out;
}

void deepseek_v4_linear_bf16_fp32_blas_(Tensor out, const Tensor &x, const Tensor &weight) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    check_blas_tensors(out, x, weight, "deepseek_v4_linear_bf16_fp32_blas_");
    if (x->size(0) <= 0 || weight->size(0) <= 0 || x->size(1) <= 0) {
        return;
    }
    launch_blas_linear_bf16_fp32(reinterpret_cast<float *>(out->data()),
                                 x->data(),
                                 weight->data(),
                                 x->size(0),
                                 weight->size(0),
                                 x->size(1));
#else
    (void)out;
    (void)x;
    (void)weight;
    throw std::runtime_error("deepseek_v4_linear_bf16_fp32_blas_ requires a HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
