#include "infinicore/ops/distributed/vocab_parallel_logits_gather.hpp"

#include "../../utils.hpp"
#include "vocab_parallel_logits_gather_kernel.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/dtype.hpp"

#include <stdexcept>
#include <string>

namespace infinicore::op::distributed {
namespace {

struct VocabParallelLogitsGatherPlannedMeta {
    graph::GraphTensor output;
    graph::GraphTensor local_logits;
    Tensor gathered;
    infinicclComm_t communicator;
    size_t tokens;
    size_t vocab_local;
    size_t world_size;
    size_t element_size;
};

size_t required_workspace_numel(const Tensor &local_logits, size_t world_size) {
    if (world_size <= 1 || local_logits->size(0) <= 1) {
        return 0;
    }
    return world_size * local_logits->size(0) * local_logits->size(1);
}

void check_vocab_parallel_logits_gather_shape(const Tensor &output,
                                              const Tensor &local_logits,
                                              const char *op_name) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, local_logits);
    if (output->ndim() != 2 || local_logits->ndim() != 2) {
        throw std::runtime_error(std::string(op_name) + " expects 2D logits tensors.");
    }
    if (!output->is_contiguous() || !local_logits->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous tensors.");
    }
    if (output->dtype() != local_logits->dtype()) {
        throw std::runtime_error(std::string(op_name) + " expects output/local_logits dtype match.");
    }
    if (output->size(0) != local_logits->size(0)) {
        throw std::runtime_error(std::string(op_name) + " token dimension mismatch.");
    }
    if (local_logits->size(1) == 0 || output->size(1) % local_logits->size(1) != 0) {
        throw std::runtime_error(std::string(op_name) + " vocab dimension mismatch.");
    }
}

Tensor prepare_workspace(Tensor workspace,
                         const Tensor &local_logits,
                         size_t world_size,
                         const char *op_name) {
    const auto required = required_workspace_numel(local_logits, world_size);
    if (required == 0) {
        return Tensor();
    }
    if (!workspace) {
        return Tensor::empty({world_size * local_logits->size(0), local_logits->size(1)},
                             local_logits->dtype(),
                             local_logits->device());
    }
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(workspace, local_logits);
    if (!workspace->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous workspace.");
    }
    if (workspace->dtype() != local_logits->dtype()) {
        throw std::runtime_error(std::string(op_name) + " expects workspace dtype to match local_logits.");
    }
    if (workspace->numel() < required) {
        throw std::runtime_error(std::string(op_name) + " workspace is too small.");
    }
    if (workspace->numel() == required) {
        return workspace->view({world_size * local_logits->size(0), local_logits->size(1)});
    }
    if (workspace->ndim() != 1) {
        throw std::runtime_error(std::string(op_name) + " expects oversized workspace to be 1D.");
    }
    return workspace->narrow({{0, 0, required}})->view({world_size * local_logits->size(0), local_logits->size(1)});
}

void run_allgather(void *output,
                   void *input,
                   size_t count,
                   DataType dtype,
                   infinicclComm_t communicator) {
    INFINICORE_CHECK_ERROR(infinicclAllGather(input,
                                              output,
                                              count,
                                              static_cast<infiniDtype_t>(static_cast<int>(dtype)),
                                              communicator,
                                              infinicore::context::getStream()));
}

} // namespace

VocabParallelLogitsGather::VocabParallelLogitsGather(Tensor output,
                                                     const Tensor &local_logits,
                                                     infinicclComm_t communicator)
    : VocabParallelLogitsGather(output, local_logits, Tensor(), communicator) {
}

VocabParallelLogitsGather::VocabParallelLogitsGather(Tensor output,
                                                     const Tensor &local_logits,
                                                     Tensor workspace,
                                                     infinicclComm_t communicator) {
    constexpr const char *op_name = "vocab_parallel_logits_gather_";
    check_vocab_parallel_logits_gather_shape(output, local_logits, op_name);

    const size_t world_size = output->size(1) / local_logits->size(1);
    Tensor gathered = prepare_workspace(workspace, local_logits, world_size, op_name);

    planned_meta_ = new VocabParallelLogitsGatherPlannedMeta{
        graph::GraphTensor(output),
        graph::GraphTensor(local_logits),
        gathered,
        communicator,
        local_logits->size(0),
        local_logits->size(1),
        world_size,
        dsize(local_logits->dtype()),
    };
}

VocabParallelLogitsGather::~VocabParallelLogitsGather() {
    if (planned_meta_) {
        auto *meta = reinterpret_cast<VocabParallelLogitsGatherPlannedMeta *>(planned_meta_);
        delete meta;
    }
}

void VocabParallelLogitsGather::run() const {
    auto *meta = reinterpret_cast<VocabParallelLogitsGatherPlannedMeta *>(planned_meta_);
    if (meta->world_size <= 1 || meta->communicator == nullptr) {
        meta->output->copy_from(meta->local_logits);
        return;
    }

    if (meta->tokens == 1) {
        run_allgather(meta->output->data(),
                      meta->local_logits->data(),
                      meta->local_logits->numel(),
                      meta->local_logits->dtype(),
                      meta->communicator);
        return;
    }

    run_allgather(meta->gathered->data(),
                  meta->local_logits->data(),
                  meta->local_logits->numel(),
                  meta->local_logits->dtype(),
                  meta->communicator);

#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    launch_vocab_parallel_logits_reorder(meta->output->data(),
                                         meta->gathered->data(),
                                         meta->tokens,
                                         meta->vocab_local,
                                         meta->world_size,
                                         meta->element_size,
                                         infinicore::context::getStream());
#else
    throw std::runtime_error("vocab_parallel_logits_gather_ multi-token reorder requires a GPU build.");
#endif
}

void VocabParallelLogitsGather::execute(Tensor output,
                                        const Tensor &local_logits,
                                        infinicclComm_t communicator) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(VocabParallelLogitsGather,
                                      output, local_logits, communicator);
}

void VocabParallelLogitsGather::execute(Tensor output,
                                        const Tensor &local_logits,
                                        Tensor workspace,
                                        infinicclComm_t communicator) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(VocabParallelLogitsGather,
                                      output, local_logits, workspace, communicator);
}

Tensor vocab_parallel_logits_gather(const Tensor &local_logits,
                                    size_t world_size,
                                    infinicclComm_t communicator) {
    if (world_size <= 1 || communicator == nullptr) {
        return local_logits;
    }
    if (local_logits->ndim() != 2) {
        throw std::runtime_error("vocab_parallel_logits_gather expects local_logits [tokens, vocab_local].");
    }

    auto shape = local_logits->shape();
    shape[1] *= world_size;
    auto output = Tensor::empty(shape, local_logits->dtype(), local_logits->device());
    vocab_parallel_logits_gather_(output, local_logits, communicator);
    return output;
}

size_t vocab_parallel_logits_gather_workspace_numel(const Tensor &local_logits,
                                                    size_t world_size) {
    if (world_size <= 1 || !local_logits || local_logits->ndim() != 2) {
        return 0;
    }
    return required_workspace_numel(local_logits, world_size);
}

void vocab_parallel_logits_gather_(Tensor output,
                                   const Tensor &local_logits,
                                   infinicclComm_t communicator) {
    VocabParallelLogitsGather::execute(output, local_logits, communicator);
}

void vocab_parallel_logits_gather_(Tensor output,
                                   const Tensor &local_logits,
                                   Tensor workspace,
                                   infinicclComm_t communicator) {
    VocabParallelLogitsGather::execute(output, local_logits, workspace, communicator);
}

} // namespace infinicore::op::distributed
