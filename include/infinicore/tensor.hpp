#pragma once

#include "device.hpp"
#include "dtype.hpp"
#include "memory.hpp"

#include <memory>
#include <vector>
namespace infinicore {

using Size = std::size_t;
using Stride = std::ptrdiff_t;
using Shape = std::vector<Size>;
using Strides = std::vector<Stride>;

class TensorImpl;

struct TensorMetaData {
    Shape shape;
    Strides strides;
    DataType dtype;
};

struct TensorData {
    size_t offset;
    std::shared_ptr<Memory> memory;
};

class Tensor {
public:
    static Tensor empty(const Shape &shape,
                        const DataType &dtype,
                        const Device &device);

    static Tensor zeros(const Shape &shape,
                        const DataType &dtype,
                        const Device &device);

    static Tensor ones(const Shape &shape,
                       const DataType &dtype,
                       const Device &device);

    Tensor(const Tensor &) = default;
    Tensor(Tensor &&) = default;
    Tensor &operator=(const Tensor &) = default;
    Tensor &operator=(Tensor &&) = default;

    TensorImpl *operator->();
    const TensorImpl *operator->() const;

protected:
    explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<TensorImpl> impl_;
    friend class TensorImpl;
};

class TensorImpl {

public:
    const Shape &shape() const;

    DataType dtype() const;

    Device device() const;

protected:
    static Tensor empty(const Shape &shape, const DataType &dtype, const Device &device);
    static Tensor zeros(const Shape &shape, const DataType &dtype, const Device &device);
    static Tensor ones(const Shape &shape, const DataType &dtype, const Device &device);

    TensorImpl();
    friend class Tensor;

private:
    TensorMetaData meta_;
    TensorData data_;
};

} // namespace infinicore
