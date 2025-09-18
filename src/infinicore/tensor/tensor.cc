#include "infinicore/tensor.hpp"

namespace infinicore {

TensorImpl *Tensor::operator->() { return impl_.get(); }

const TensorImpl *Tensor::operator->() const { return impl_.get(); }

const Shape &TensorImpl::shape() const {
    return meta_.shape;
}

DataType TensorImpl::dtype() const {
    return meta_.dtype;
}

Device TensorImpl::device() const {
    return data_.memory->device();
}
} // namespace infinicore
