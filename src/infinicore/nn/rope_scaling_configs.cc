#include "infinicore/nn/rope_scaling_configs.hpp"
#include <cmath>
#include <stdexcept>

namespace infinicore::nn {

// LongRopeScalingConfig Implementation
LongRopeScalingConfig::LongRopeScalingConfig(
    std::vector<float> short_factor,
    std::vector<float> long_factor,
    size_t original_max_position_embeddings,
    float factor)
    : short_factor_(std::move(short_factor)),
      long_factor_(std::move(long_factor)),
      original_max_position_embeddings_(original_max_position_embeddings),
      factor_(factor == 1.0f ? 1.0f : std::sqrt(1 + std::log(factor) / std::log(original_max_position_embeddings))) {}

float LongRopeScalingConfig::get_freq_scale(size_t pos, size_t dim_idx, float base_inv_freq) const {
    float _ext = (pos < original_max_position_embeddings_) ? short_factor_[dim_idx] : long_factor_[dim_idx];
    // The base inv_freq is multiplied by this scale.
    // Original: inv_freq = 1.0f / (_ext * pow(theta, 2j/head_dim))
    // New: inv_freq = base_inv_freq * (1.0f / _ext)
    return 1.0f / _ext;
}

float LongRopeScalingConfig::get_magnitude_scale(size_t pos, size_t dim_idx, float base_inv_freq) const {
    return factor_;
}

// TODO(rubik) llama3 implement here
// Llama3RopeScalingConfig Implementation
Llama3RopeScalingConfig::Llama3RopeScalingConfig(
    float factor,
    float low_freq_factor,
    float high_freq_factor,
    size_t original_max_position_embeddings)
    : factor_(factor),
      low_freq_factor_(low_freq_factor),
      high_freq_factor_(high_freq_factor),
      original_max_position_embeddings_(original_max_position_embeddings) {}

float Llama3RopeScalingConfig::get_freq_scale(size_t pos, size_t dim_idx, float base_inv_freq) const {
    return 1.0f;
}

} // namespace infinicore::nn
