#include "tensor.hpp"

namespace infinicore::py {

Tensor::Tensor(const infinicore::Tensor &tensor) : tensor_{tensor} {}

Tensor empty(const Shape &shape,
             const DataType &dtype,
             const Device &device,
             bool pin_memory) {
    return Tensor{infinicore::Tensor::empty(shape, dtype, device, pin_memory)};
}

} // namespace infinicore::py
