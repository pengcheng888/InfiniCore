#ifndef __TOPKSOFTMAX_INFO_H__
#define __TOPKSOFTMAX_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::topksoftmax {

class TopksoftmaxInfo {
    TopksoftmaxInfo() = default;

public:
    infiniDtype_t xtype;
    infiniDtype_t vtype;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> y_strides;
    std::vector<ptrdiff_t> x_strides;
    //
    int N;
    int width;
    int topk;

public:
    size_t ndim() const { return shape.size(); }
    size_t dim() const { return shape[ndim() - 1]; }

    static utils::Result<TopksoftmaxInfo> create(
        infiniopTensorDescriptor_t values_desc,
        infiniopTensorDescriptor_t indices_desc,
        infiniopTensorDescriptor_t x_desc) {

        auto vtype = values_desc->dtype();
        auto xtype = x_desc->dtype();
        if (xtype != vtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        // ....... 其他的判断
        if (values_desc->ndim() != 2 || indices_desc->ndim() != 2 || x_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        size_t batch = x_desc->shape()[0];
        int N = batch;
        int width = x_desc->shape()[1];
        int topk = indices_desc->shape()[1];

        if (indices_desc->shape()[0] != batch || x_desc->shape()[0] != batch) {

            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        return utils::Result<TopksoftmaxInfo>(TopksoftmaxInfo{
            xtype,
            vtype,
            values_desc->shape(),
            values_desc->strides(),
            indices_desc->strides(),
            N,
            width,
            topk,
        });
    }
};

} // namespace op::topksoftmax

#endif // __TOPKSOFTMAX_INFO_H__
