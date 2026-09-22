#include "int8_gemm_metax.h"

#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

#include <algorithm>
#include <cstddef>

namespace op::i8gemm::metax {
namespace {

constexpr int kMinimumInt8GemmRows = 17;

bool is_contiguous_matrix(
    infiniopTensorDescriptor_t desc,
    size_t rows,
    size_t cols) {
    return desc->ndim() == 2
        && desc->dim(0) == rows
        && desc->dim(1) == cols
        && static_cast<size_t>(desc->stride(0)) == cols
        && static_cast<size_t>(desc->stride(1)) == 1;
}

bool is_contiguous_vector(
    infiniopTensorDescriptor_t desc,
    size_t rows) {
    return (desc->ndim() == 1
            && desc->dim(0) == rows
            && desc->stride(0) == 1)
        || (desc->ndim() == 2
            && desc->dim(0) == rows
            && desc->dim(1) == 1
            && desc->stride(0) == 1
            && desc->stride(1) == 1);
}

bool is_transposed_packed_weights(
    infiniopTensorDescriptor_t desc,
    size_t rows,
    size_t cols) {
    return desc->ndim() == 2
        && desc->dim(0) == rows
        && desc->dim(1) == cols
        && static_cast<size_t>(desc->stride(0)) == 1
        && static_cast<size_t>(desc->stride(1)) == rows;
}

bool is_supported_packed_weights(
    infiniopTensorDescriptor_t desc,
    size_t rows,
    size_t cols) {
    return is_contiguous_matrix(desc, rows, cols)
        || is_transposed_packed_weights(desc, rows, cols);
}

bool is_supported_bias(
    infiniopTensorDescriptor_t desc,
    infiniDtype_t dtype,
    size_t rows,
    size_t cols) {
    if (desc == nullptr) {
        return true;
    }
    if (desc->dtype() != dtype) {
        return false;
    }
    if (desc->ndim() == 1) {
        return desc->dim(0) == cols && desc->stride(0) == 1;
    }
    return desc->ndim() == 2
        && desc->dim(0) == rows
        && desc->dim(1) == cols
        && desc->stride(0) == 0
        && desc->stride(1) == 1;
}

size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

size_t padded_activation_bytes(const I8GemmInfo &info) {
    if (info.m >= kMinimumInt8GemmRows) {
        return 0;
    }
    return align_up(
        static_cast<size_t>(kMinimumInt8GemmRows) * static_cast<size_t>(info.k),
        16);
}

size_t packed_output_elements(const I8GemmInfo &info) {
    const size_t rows = std::max(
        static_cast<size_t>(info.m),
        static_cast<size_t>(kMinimumInt8GemmRows));
    return rows * static_cast<size_t>(info.n);
}

} // namespace

infiniStatus_t launchPadActivation(
    void *padded,
    const void *activation,
    int rows,
    int padded_rows,
    int cols,
    void *stream);

infiniStatus_t launchEpilogue(
    void *out,
    const void *packed_out,
    const void *bias,
    const void *a_scale,
    const void *b_scale,
    int rows,
    int cols,
    infiniDtype_t dtype,
    void *stream);

struct Descriptor::Opaque {
    std::shared_ptr<device::metax::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t bias_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t a_scale_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t b_scale_desc) {
    CHECK_DTYPE(
        out_desc->dtype(),
        INFINI_DTYPE_F16,
        INFINI_DTYPE_BF16);
    CHECK_DTYPE(a_desc->dtype(), INFINI_DTYPE_I8);
    CHECK_DTYPE(b_desc->dtype(), INFINI_DTYPE_I8);
    CHECK_DTYPE(a_scale_desc->dtype(), INFINI_DTYPE_F32);
    CHECK_DTYPE(b_scale_desc->dtype(), INFINI_DTYPE_F32);

    auto result = I8GemmInfo::create(
        out_desc, a_desc, b_desc, MatrixLayout::COL_MAJOR);
    CHECK_RESULT(result);
    const auto &info = *result;

    if (info.batch != 1
        || !is_contiguous_matrix(a_desc, info.m, info.k)
        || !is_supported_packed_weights(b_desc, info.k, info.n)
        || !is_contiguous_matrix(out_desc, info.m, info.n)
        || !is_contiguous_vector(a_scale_desc, info.m)
        || !is_contiguous_vector(b_scale_desc, info.n)) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    if (!is_supported_bias(
            bias_desc,
            out_desc->dtype(),
            static_cast<size_t>(info.m),
            static_cast<size_t>(info.n))) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    const size_t workspace_size =
        padded_activation_bytes(info)
        + packed_output_elements(info) * sizeof(int32_t);
    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    *desc_ptr = new Descriptor(
        new Opaque{handle->internal()},
        result.take(),
        workspace_size,
        out_desc->dtype(),
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *bias,
    const void *a,
    const void *a_scale,
    const void *b,
    const void *b_scale,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto stream_handle = reinterpret_cast<hcStream_t>(stream);
    auto workspace_bytes = reinterpret_cast<std::byte *>(workspace);
    const auto activation = static_cast<const int8_t *>(a);
    int8_t *gemm_activation = nullptr;
    int32_t *packed_out = nullptr;
    size_t gemm_rows = static_cast<size_t>(_info.m);

    if (_info.m < kMinimumInt8GemmRows) {
        gemm_rows = kMinimumInt8GemmRows;
        gemm_activation = reinterpret_cast<int8_t *>(workspace_bytes);
        packed_out = reinterpret_cast<int32_t *>(
            workspace_bytes + padded_activation_bytes(_info));

        CHECK_STATUS(launchPadActivation(
            gemm_activation,
            activation,
            _info.m,
            static_cast<int>(gemm_rows),
            _info.k,
            stream_handle));
    } else {
        gemm_activation = const_cast<int8_t *>(activation);
        packed_out = reinterpret_cast<int32_t *>(workspace_bytes);
    }

    int32_t alpha = 1;
    int32_t beta = 0;
    const bool transposed_weights = _info.b_matrix.row_stride == 1;
    const auto weights_op = transposed_weights ? HCBLAS_OP_T : HCBLAS_OP_N;
    const int weights_ld = static_cast<int>(_info.b_matrix.ld());
    CHECK_STATUS(_opaque->internal->useMcblas(
        stream_handle,
        [&](hcblasHandle_t blas_handle) {
            CHECK_MCBLAS(hcblasGemmEx(
                blas_handle,
                weights_op,
                HCBLAS_OP_N,
                _info.n,
                static_cast<int>(gemm_rows),
                _info.k,
                &alpha,
                b,
                HPCC_R_8I,
                weights_ld,
                gemm_activation,
                HPCC_R_8I,
                _info.k,
                &beta,
                packed_out,
                HPCC_R_32I,
                _info.n,
                HCBLAS_COMPUTE_32I,
                HCBLAS_GEMM_DEFAULT));
            return INFINI_STATUS_SUCCESS;
        }));

    return launchEpilogue(
        out,
        packed_out,
        bias,
        a_scale,
        b_scale,
        _info.m,
        _info.n,
        _out_dtype,
        stream_handle);
}

} // namespace op::i8gemm::metax
