#include "tensor.hpp"

namespace infinicore::py {

Tensor::Tensor(const infinicore::Tensor &tensor) : tensor_{tensor} {}

Tensor Tensor::to(const Device &device) const {
    return Tensor{tensor_->to(device)};
}

void Tensor::copy_(const Tensor &src) {
    return tensor_->copy_from(src.tensor_);
}

Tensor Tensor::permute(const Shape &shape) {
    return Tensor{tensor_->permute(shape)};
}

Tensor empty(const Shape &shape,
             const DataType &dtype,
             const Device &device,
             bool pin_memory) {
    return Tensor{infinicore::Tensor::empty(shape, dtype, device, pin_memory)};
}

Tensor from_blob(uintptr_t raw_ptr,
                 Shape &shape,
                 const DataType &dtype,
                 const Device &device) {
    return Tensor{infinicore::Tensor::from_blob(reinterpret_cast<void *>(raw_ptr), shape, dtype, device)};
}

} // namespace infinicore::py
