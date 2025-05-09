#ifndef __SIDMOID_CUDA_H__
#define __SIDMOID_CUDA_H__

#include "../../../elementwise/cuda/elementwise_cuda.cuh"
#include <cuda_fp16.h>

namespace op::sigmoid::cuda {
typedef struct SigmoidOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        // Prevent data overflow
        //         x = __hmax(__hmin(x, __float2half(10.0f)), __float2half(-10.0f))
        // sigmoid(x) = 1 / (1 + exp(-x))
        if constexpr (std::is_same_v<T, half2>) {
            half2 denominator = __hadd2(make_half2(1, 1), h2exp(__hneg2(x)));
            return h2rcp(denominator);
        } else if constexpr (std::is_same_v<T, half>) {
            half denominator = __hadd(__float2half(1.0f), hexp(__hneg(x)));
            return hrcp(denominator);
        } else if constexpr (std::is_same_v<T, float>) {
            float denominator = __fadd_rn(1.0f, __expf(-x));
            return __frcp_rn(denominator);
        } else { // double
            return 1.0 / (1.0 + exp(-x));
        }
    }
} SigmoidOp;
} // namespace op::sigmoid::cuda

#endif // __SIDMOID_CUDA_H__
