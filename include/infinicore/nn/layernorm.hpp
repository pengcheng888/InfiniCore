#pragma once

#include "../ops.hpp"
#include "module.hpp"

namespace infinicore::nn {

class LayerNorm : public Module {
public:
    /**
     * @brief Construct a LayerNorm layer
     *
     * @param normalized_shape Size of the feature dimension to normalize (typically hidden_size)
     * @param eps Small constant for numerical stability (default: 1e-6)
     * @param dtype Data type for the weight (default: DataType::F32)
     * @param device Device to create the weight on
     * @param elementwise_affine Whether to include learnable affine weight and bias parameters (default: true)
     */
    LayerNorm(size_t normalized_shape,
              double eps = 1e-6,
              const DataType &dtype = DataType::F32,
              const Device &device = Device(),
              bool elementwise_affine = true);

    /**
     * @brief Forward pass: apply LayerNorm
     *
     * @param x Input tensor of shape (*, normalized_shape) where * is any number of dimensions
     * @return Normalized tensor with same shape as input
     *
     * The normalization is applied over the last dimension.
     * For example:
     *   Input: [batch, seq_len, hidden_size] -> normalize over hidden_size
     *   Input: [batch, hidden_size] -> normalize over hidden_size
     */
    Tensor forward(const Tensor &x) const;

    double eps() const { return eps_; }
    DataType dtype() const { return dtype_; }

    // String representation
    std::string extra_repr() const;

    // Accessors for parameters
    Tensor weight() const { return weight_; }
    Tensor bias() const { return bias_; }

protected:
    // Parameters
    INFINICORE_NN_PARAMETER(weight);
    INFINICORE_NN_PARAMETER(bias);

private:
    size_t normalized_shape_; // Size of the feature dimension
    double eps_;              // Epsilon for numerical stability
    DataType dtype_;          // Data type for weight
    bool elementwise_affine_; // Whether to use learnable affine parameters
};
} // namespace infinicore::nn
