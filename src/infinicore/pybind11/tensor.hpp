#include "infinicore.hpp"

#include "device.hpp"
#include "dtype.hpp"

namespace infinicore::py {

class Tensor {
public:
    Tensor(const infinicore::Tensor &tensor);

private:
    infinicore::Tensor tensor_;
};

Tensor empty(const Shape &shape,
             const DataType &dtype,
             const Device &device,
             bool pin_memory = false);

} // namespace infinicore::py
