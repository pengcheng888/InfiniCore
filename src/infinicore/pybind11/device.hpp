#include "infinicore.hpp"

namespace infinicore::py {

class Device {
public:
    using Index = infinicore::Device::Index;

    Device(const infinicore::Device &device);

    Device(const std::string &type, Index index);

    const std::string &getType() const;

    const Index &getIndex() const;

    std::string toRepresentation() const;

    std::string toString() const;

private:
    std::string type_;

    Index index_;

    infinicore::Device device_;
};

} // namespace infinicore::py
