#include "infinicore/ops/add.hpp"
#include "infinicore/ops/linear.hpp"
#include "infinicore/ops/matmul.hpp"
#include "infinicore/ops/rearrange.hpp"

namespace infinicore::op {

Tensor embedding(Tensor input, // LongTensor of arbitrary shape containing the indices to extract
                 Tensor weight // Weight: Embedding matrix of floating point type with shape (V, embedding_dim), where V = maximum index + 1
) {
    assert(infinicore::DataType::I64 == input->dtype() || (infinicore::DataType::I32 == input->dtype()));

    auto input_shape = input->shape();
    auto weight_shape = weight->shape();

    auto vocab_size = weight_shape[0];
    auto embedding_dim = weight_shape[1];

    auto batch_size = input_shape[0];
    auto ntoken = input_shape[1];

    Tensor inputs_embeds = Tensor::empty({batch_size, ntoken, embedding_dim}, weight->dtype(), weight->device());

    const Size counts = batch_size * ntoken;
    const Size bytes = dsize(weight->dtype()) * embedding_dim;

    if (infinicore::DataType::I64 == input->dtype()) {
        for (Size i = 0; i < counts; ++i) {
            int id = *(int64_t *)input->data(i);
            assert((id >= 0) && (id < vocab_size));
            infinirtMemcpyAsync(inputs_embeds->data(i * embedding_dim),
                                weight->data(id * embedding_dim),
                                bytes,
                                INFINIRT_MEMCPY_D2D,
                                nullptr);
        }
    } else if (infinicore::DataType::I32 == input->dtype()) {
        for (Size i = 0; i < counts; ++i) {
            int id = *(int32_t *)input->data(i);
            assert((id >= 0) && (id < vocab_size));
            infinirtMemcpyAsync(inputs_embeds->data(i * embedding_dim),
                                weight->data(id * embedding_dim),
                                bytes,
                                INFINIRT_MEMCPY_D2D,
                                nullptr);
        }
    }

    return inputs_embeds;
}

} // namespace infinicore::op
