#include "infinicore/tensor.hpp"

namespace infinicore {

TensorImpl *Tensor::operator->() { return impl_.get(); }

const TensorImpl *Tensor::operator->() const { return impl_.get(); }

Tensor Tensor::empty(const Shape &shape,
                     const DataType &dtype,
                     const Device &device) {
    return Tensor{TensorImpl::empty(shape, dtype, device)};
}

Tensor Tensor::zeros(const Shape &shape,
                     const DataType &dtype,
                     const Device &device) {
    return Tensor{TensorImpl::zeros(shape, dtype, device)};
}

Tensor Tensor::ones(const Shape &shape,
                    const DataType &dtype,
                    const Device &device) {
    return Tensor{TensorImpl::ones(shape, dtype, device)};
}

const Shape &TensorImpl::shape() const {
    return meta_.shape;
}

DataType TensorImpl::dtype() const {
    return meta_.dtype;
}

Device TensorImpl::device() const {
    return data_.memory->device();
}

std::shared_ptr<TensorImpl> TensorImpl::empty(const Shape &shape,
                                              const DataType &dtype,
                                              const Device &device) {
    // TODO: Implement this.
    return nullptr;
}
std::shared_ptr<TensorImpl> TensorImpl::zeros(const Shape &shape,
                                              const DataType &dtype,
                                              const Device &device) {
    // TODO: Implement this.
    return nullptr;
}
std::shared_ptr<TensorImpl> TensorImpl::ones(const Shape &shape,
                                             const DataType &dtype,
                                             const Device &device) {
    // TODO: Implement this.
    return nullptr;
}

} // namespace infinicore
