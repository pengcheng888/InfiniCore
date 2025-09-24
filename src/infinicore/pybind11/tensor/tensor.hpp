#pragma once
#include "infinicore.hpp"

#include "../device/device.hpp"
#include "../dtype/dtype.hpp"

namespace infinicore::py {

class Tensor {
public:
    Tensor(const infinicore::Tensor &tensor);
    Tensor to(const Device &device) const;
    void copy_(const Tensor &src);

private:
    infinicore::Tensor tensor_;
};

Tensor empty(const Shape &shape,
             const DataType &dtype,
             const Device &device,
             bool pin_memory = false);

Tensor from_blob(uintptr_t raw_ptr,
                 Shape &shape,
                 const DataType &dtype,
                 const Device &device);

} // namespace infinicore::py
