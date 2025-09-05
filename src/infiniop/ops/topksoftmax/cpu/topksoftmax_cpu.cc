#include "topksoftmax_cpu.h"
#include "../../../../utils.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"
#include "topksoftmax_cpu.h"
#include <algorithm>

namespace op::topksoftmax::cpu {

Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc) {

    auto result = TopksoftmaxInfo::create(x_desc);
    CHECK_RESULT(result);

    auto info = result.take();
    if (info.x_strides[1] != 1) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    *desc_ptr = new Descriptor(nullptr,
                               std::move(info),
                               0,
                               handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t topksoftmax_cpu_float(float *values, int *indices,
                                     const float *x, // 先假定是 float
                                     size_t topk, bool norm, size_t N, size_t width) {
    /*
    O-----------> width
    |
    |
    N
    */

    // printf("norm:  %d \n", norm);
    // printf(" N:  %ld \n", N);
    // printf(" topk:  %ld \n", topk);
    // printf(" width:  %ld \n", width);
    // printf("\n");
    // for (size_t i = 0; i < width; ++i) {
    //     printf("  %f ", x[i]);
    // }
    // printf("\n");

    //  _f16_to_f32(fp16_t val)

    for (size_t n = 0; n < N; ++n) {
        const float *x_input = x + n * width;
        float *values_input = values + n * topk;
        int *indices_input = indices + n * topk;

        std::vector<std::pair<float, size_t>> value_index_arr;
        value_index_arr.resize(width);

        // ------------------------------------------------ //
        //             第一步：计算最大值                       //
        // ------------------------------------------------ //
        float value_max = x_input[0];
        for (size_t i = 1; i < width; ++i) {
            value_max = x_input[i] > value_max ? x_input[i] : value_max;
        }
        // printf("value_max: %f \n", value_max);

        // ------------------------------------------------ //
        //             第二步：计算指数和                      //
        // ------------------------------------------------ //
        float exp_sum = 0.0f;
        for (size_t i = 0; i < width; ++i) {
            float value = std::exp(x_input[i] - value_max);
            value_index_arr[i] = {value, i};
            exp_sum += value;
        }
        // printf("exp_sum: %f \n", exp_sum);

        // ------------------------------------------------ //
        //              第三步：计算 Softmax                  //
        // ------------------------------------------------ //
        for (size_t i = 0; i < width; ++i) {
            value_index_arr[i].first /= exp_sum;
        }

        // ------------------------------------------------ //
        //           第四步：计算 排序                        //
        // ------------------------------------------------ //
        std::sort(value_index_arr.begin(), value_index_arr.end(),
                  [](const std::pair<float, size_t> &a, const std::pair<float, size_t> &b) {
                      return a.first > b.first;
                  });

        // printf("\n");
        // for (size_t i = 0; i < width; ++i) {
        //     printf("  %f ", value_index_arr[i].first);
        // }
        // printf("\n");
        // ------------------------------------------------ //
        //           第五步： 获取topk                        //
        // ------------------------------------------------ //
        exp_sum = 0.0f;
        for (size_t i = 0; i < topk; ++i) {
            float value = value_index_arr[i].first;
            exp_sum += value;

            values_input[i] = value;
            indices_input[i] = value_index_arr[i].second;
        }

        // ------------------------------------------------ //
        //           第6步： norm归一化                       //
        // ------------------------------------------------ //
        if (norm) {
            for (size_t i = 0; i < topk; ++i) {
                values_input[i] /= exp_sum;
            }
        }
    }

    return INFINI_STATUS_SUCCESS;
}

template <typename T>
infiniStatus_t topksoftmax_cpu_fp16_bf16(float *values, int *indices,
                                         const T *x, // 先假定是 float
                                         size_t topk, bool norm, size_t N, size_t width) {
    /*
    O-----------> width
    |
    |
    N
    */

    for (size_t n = 0; n < N; ++n) {
        const T *x_input = x + n * width;
        float *values_input = values + n * topk;
        int *indices_input = indices + n * topk;

        std::vector<std::pair<float, size_t>> value_index_arr;
        value_index_arr.resize(width);

        // ------------------------------------------------ //
        //             第0步： 数据先转换到 float              //
        // ------------------------------------------------ //
        for (size_t i = 0; i < width; ++i) {
            value_index_arr[i].second = i;

            if constexpr (std::is_same<T, fp16_t>::value) {
                value_index_arr[i].first = _f16_to_f32(x_input[i]);
            } else if constexpr (std::is_same<T, bf16_t>::value) {
                value_index_arr[i].first = _bf16_to_f32(x_input[i]);
            } else {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
        }
        // ------------------------------------------------ //
        //             第一步：计算最大值                       //
        // ------------------------------------------------ //
        float value_max = value_index_arr[0].first;
        for (size_t i = 1; i < width; ++i) {
            value_max = value_index_arr[i].first > value_max ? value_index_arr[i].first : value_max;
        }
        // printf("value_max: %f \n", value_max);

        // ------------------------------------------------ //
        //             第二步：计算指数和                      //
        // ------------------------------------------------ //
        float exp_sum = 0.0f;
        for (size_t i = 0; i < width; ++i) {
            float value = std::exp(value_index_arr[i].first - value_max);
            exp_sum += value;
            value_index_arr[i].first = value;
        }
        // printf("exp_sum: %f \n", exp_sum);

        // ------------------------------------------------ //
        //              第三步：计算 Softmax                  //
        // ------------------------------------------------ //
        for (size_t i = 0; i < width; ++i) {
            value_index_arr[i].first /= exp_sum;
        }

        // ------------------------------------------------ //
        //           第四步：计算 排序                        //
        // ------------------------------------------------ //
        std::sort(value_index_arr.begin(), value_index_arr.end(),
                  [](const std::pair<float, size_t> &a, const std::pair<float, size_t> &b) {
                      return a.first > b.first;
                  });

        // printf("\n");
        // for (size_t i = 0; i < width; ++i) {
        //     printf("  %f ", value_index_arr[i].first);
        // }
        // printf("\n");
        // ------------------------------------------------ //
        //           第五步： 获取topk                        //
        // ------------------------------------------------ //
        exp_sum = 0.0f;
        for (size_t i = 0; i < topk; ++i) {
            float value = value_index_arr[i].first;
            exp_sum += value;

            values_input[i] = value;
            indices_input[i] = value_index_arr[i].second;
        }

        // ------------------------------------------------ //
        //           第6步： norm归一化                       //
        // ------------------------------------------------ //
        if (norm) {
            for (size_t i = 0; i < topk; ++i) {
                values_input[i] /= exp_sum;
            }
        }
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    float *values, int *indices, void *x,
    size_t topk, bool norm,
    void *stream) const {

    size_t N = _info.N;
    size_t width = _info.width;

    if (_info.xtype == INFINI_DTYPE_F32) {
        topksoftmax_cpu_float(values, indices, (const float *)x, topk, norm, N, width);
    } else if (_info.xtype == INFINI_DTYPE_F16) {
        topksoftmax_cpu_fp16_bf16<fp16_t>(values, indices, (const fp16_t *)x, topk, norm, N, width);
    } else if (_info.xtype == INFINI_DTYPE_BF16) {
        topksoftmax_cpu_fp16_bf16<bf16_t>(values, indices, (const bf16_t *)x, topk, norm, N, width);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::topksoftmax::cpu
