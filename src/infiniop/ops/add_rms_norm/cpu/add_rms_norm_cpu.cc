#include "add_rms_norm_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"

namespace op::add_rms_norm::cpu {

Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t weight_desc,
    float epsilon) {
    auto result = AddRMSNormInfo::create(y_desc, a_desc, b_desc, weight_desc, epsilon);
    CHECK_RESULT(result);
    *desc_ptr = new Descriptor(nullptr, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
infiniStatus_t add_rmsnorm(const AddRMSNormInfo *info, T *y, const T *a, const T *b, const T *w) {
    const size_t batch_size = info->shape[0];
    const size_t nhead = info->ndim() > 2 ? info->shape[1] : 1;
    const size_t dim = info->dim();
    const ptrdiff_t total_blocks = static_cast<ptrdiff_t>(batch_size * nhead);

#pragma omp parallel for
    for (ptrdiff_t block_idx = 0; block_idx < total_blocks; ++block_idx) {
        const size_t i = block_idx / nhead; // batch index
        const size_t j = block_idx % nhead; // head index

        const T *a_ptr = a + i * info->a_strides[0] + j * info->a_strides[1];
        const T *b_ptr = b + i * info->b_strides[0] + j * info->b_strides[1];
        T *y_ptr = y + i * info->y_strides[0] + j * info->y_strides[1];

        // First, compute add(a, b) and store sum values
        // We'll compute RMS norm directly on the sum
        T sum_squared = (T)0;
        for (size_t k = 0; k < dim; k++) {
            T sum_val = a_ptr[k] + b_ptr[k];
            sum_squared += sum_val * sum_val;
        }

        // Compute RMS: 1 / (sqrt(mean(sum^2) + eps))
        // Note: mean = sum_squared / dim
        T rms = (T)1 / std::sqrt(sum_squared / (T)(dim) + (T)(info->epsilon));

        // Apply normalization: y = (a + b) * w * rms
        // Recompute sum to avoid storing temporary array
        for (size_t k = 0; k < dim; k++) {
            T sum_val = a_ptr[k] + b_ptr[k];
            y_ptr[k] = sum_val * w[k] * rms;
        }
    }

    return INFINI_STATUS_SUCCESS;
}

template <typename T, typename Tw>
infiniStatus_t add_rmsnormHalfPrecision(const AddRMSNormInfo *info, T *y, const T *a, const T *b, const Tw *w) {
    static_assert(std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value,
                  "T must be fp16_t or bf16_t");

    const size_t batch_size = info->shape[0];
    const size_t nhead = info->ndim() > 2 ? info->shape[1] : 1;
    const size_t dim = info->dim();
    const ptrdiff_t total_blocks = static_cast<ptrdiff_t>(batch_size * nhead);

#pragma omp parallel for
    for (ptrdiff_t block_idx = 0; block_idx < total_blocks; ++block_idx) {
        const size_t i = block_idx / nhead; // batch index
        const size_t j = block_idx % nhead; // head index

        const T *a_ptr = a + i * info->a_strides[0] + j * info->a_strides[1];
        const T *b_ptr = b + i * info->b_strides[0] + j * info->b_strides[1];
        T *y_ptr = y + i * info->y_strides[0] + j * info->y_strides[1];

        // Compute sum of squares for RMS normalization
        float sum_squared = 0.0f;
        for (size_t k = 0; k < dim; k++) {
            float sum_val = utils::cast<float>(a_ptr[k]) + utils::cast<float>(b_ptr[k]);
            sum_squared += sum_val * sum_val;
        }

        // Compute RMS: 1 / (sqrt(sum/dim + eps))
        float rms = 1.f / std::sqrt(sum_squared / (float)(dim) + info->epsilon);

        // Apply normalization: y = (a + b) * w * rms
        for (size_t k = 0; k < dim; k++) {
            float sum_val = utils::cast<float>(a_ptr[k]) + utils::cast<float>(b_ptr[k]);
            float val;
            if constexpr (std::is_same<Tw, float>::value) {
                val = sum_val * w[k] * rms;
            } else if constexpr (std::is_same<Tw, T>::value || std::is_same_v<Tw, fp16_t> || std::is_same_v<Tw, bf16_t>) {
                val = sum_val * utils::cast<float>(w[k]) * rms;
            } else {
                std::abort();
            }
            y_ptr[k] = utils::cast<T>(val);
        }
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y, const void *a, const void *b, const void *weight,
    void *stream) const {
    if (_info.atype == INFINI_DTYPE_F16) {
        if (_info.wtype == INFINI_DTYPE_F16) {
            CHECK_STATUS(add_rmsnormHalfPrecision(&_info, (fp16_t *)y, (const fp16_t *)a, (const fp16_t *)b, (const fp16_t *)weight));
        } else if (_info.wtype == INFINI_DTYPE_F32) {
            CHECK_STATUS(add_rmsnormHalfPrecision(&_info, (fp16_t *)y, (const fp16_t *)a, (const fp16_t *)b, (const float *)weight));
        } else if (_info.wtype == INFINI_DTYPE_BF16) {
            CHECK_STATUS(add_rmsnormHalfPrecision(&_info, (fp16_t *)y, (const fp16_t *)a, (const fp16_t *)b, (const bf16_t *)weight));
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (_info.atype == INFINI_DTYPE_BF16) {
        if (_info.wtype == INFINI_DTYPE_BF16) {
            CHECK_STATUS(add_rmsnormHalfPrecision(&_info, (bf16_t *)y, (const bf16_t *)a, (const bf16_t *)b, (const bf16_t *)weight));
        } else if (_info.wtype == INFINI_DTYPE_F32) {
            CHECK_STATUS(add_rmsnormHalfPrecision(&_info, (bf16_t *)y, (const bf16_t *)a, (const bf16_t *)b, (const float *)weight));
        } else if (_info.wtype == INFINI_DTYPE_F16) {
            CHECK_STATUS(add_rmsnormHalfPrecision(&_info, (bf16_t *)y, (const bf16_t *)a, (const bf16_t *)b, (const fp16_t *)weight));
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (_info.atype == INFINI_DTYPE_F32) {
        CHECK_STATUS(add_rmsnorm(&_info, (float *)y, (const float *)a, (const float *)b, (const float *)weight));
    } else if (_info.atype == INFINI_DTYPE_F64) {
        CHECK_STATUS(add_rmsnorm(&_info, (double *)y, (const double *)a, (const double *)b, (const double *)weight));
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::add_rms_norm::cpu
