#ifndef __ZEROS_CPU_H__
#define __ZEROS_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(zeros, cpu)

namespace op::zeros::cpu {
typedef struct ZerosOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x) const {
        return 0.0f;
    }
} ZerosOp;
} // namespace op::zeros::cpu

#endif // __ZEROS_CPU_H__
