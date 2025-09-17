#include "infinicore/tensor.hpp"

namespace infinicore {

const Shape &TensorImpl::shape() const {
    return _meta.shape;
}

DataType TensorImpl::dtype() const {
    return _meta.dtype;
}

Device TensorImpl::device() const {
    return _data.storage->device();
}
} // namespace infinicore
