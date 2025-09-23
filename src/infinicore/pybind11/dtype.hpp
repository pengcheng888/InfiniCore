#include "infinicore.hpp"

namespace infinicore::py {

class DataType {
public:
    DataType(const infinicore::DataType &dtype);

    static std::string toString(const DataType &dtype);

private:
    infinicore::DataType dtype_;
};

} // namespace infinicore::py
