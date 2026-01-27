#ifndef __CONV_NINETOOTHED_DESCRIPTOR_H__
#define __CONV_NINETOOTHED_DESCRIPTOR_H__

#include "../../../handle.h"
#include "../../../operator.h"
#include "../../../tensor.h"

#include "../../../../../build/ninetoothed/conv2d.h"
#include "../../../ninetoothed/utils.h"

#include "../../../operator.h"
#include "../../conv/info.h"

namespace op::conv::ninetoothed {

class Descriptor final : public InfiniopDescriptor {
    struct Opaque;
    Opaque *_opaque;
    infiniDtype_t _dtype;
    ConvInfo _info;
    size_t _workspace_size;

    Descriptor(
        infiniDtype_t dtype,
        ConvInfo info,
        size_t workspace_size_,
        Opaque *opaque,
        infiniDevice_t device_type,
        int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _opaque(opaque),
          _dtype(dtype),
          _info(info),
          _workspace_size(workspace_size_) {}

public:
    ~Descriptor() {
    }

    size_t workspaceSize() const { return _workspace_size; }

    static infiniStatus_t create(
        infiniopHandle_t handle_,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t w_desc,
        infiniopTensorDescriptor_t b_desc,
        const void *pads,
        const void *strides,
        const void *dilations,
        size_t n) {

        // auto handle = handle_;
        // auto dtype = y_desc->dtype();

        // CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

        // auto result = ConvInfo::create(handle_, y_desc, x_desc, w_desc, b_desc,
        //                                pads, strides, dilations, n);

        // CHECK_RESULT(result);
        // auto conv_info = result.take();
        // auto opaque_result = Opaque::create(handle->internal(), conv_info, dtype);
        // CHECK_RESULT(opaque_result);
        // auto opaque = new Opaque(opaque_result.take());

        // *desc_ptr = new Descriptor(
        //     dtype,
        //     std::move(conv_info),
        //     opaque->workspace_size,
        //     opaque,
        //     handle->device,
        //     handle->device_id);
        return INFINI_STATUS_NOT_IMPLEMENTED;
        // return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t calculate(
        void *workspace, size_t workspace_size,
        void *y,
        const void *x,
        const void *w,
        const void *bias,
        void *stream) const {

        // auto out_nt{::ninetoothed::Tensor(output, out_shape_, out_strides_)};
        // auto up_nt{::ninetoothed::Tensor(inputs[0], up_shape_, up_strides_)};
        // auto gate_nt{::ninetoothed::Tensor(inputs[1], gate_shape_, gate_strides_)};

        // NineToothedResult launch_conv2d(NineToothedStream stream, NineToothedTensor input, NineToothedTensor weight, NineToothedTensor bias, NineToothedTensor output, NineToothedTensor input_precision, int input_precision_, int stride_h_, int stride_w_, int padding_h_, int padding_w_, int dilation_h_, int dilation_w_, int dtype_, int block_size_m_, int block_size_n_, int block_size_k_);

        // if (launch_swiglu(stream,
        //                   out_nt,
        //                   up_nt,
        //                   gate_nt,
        //                   out_shape_.size(),
        //                   dtype_,
        //                   1024)) {
        //     return INFINI_STATUS_NOT_IMPLEMENTED;
        // }
        return INFINI_STATUS_NOT_IMPLEMENTED;
        // return INFINI_STATUS_SUCCESS;
    }
};

} // namespace op::conv::ninetoothed

#endif // __CONV_NINETOOTHED_DESCRIPTOR_H__