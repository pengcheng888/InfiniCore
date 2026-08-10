#pragma once

#include "../../device.hpp"
#include "../../graph/graph.hpp"
#include "../common/op.hpp"

#include <infiniccl.h>

namespace infinicore::op::distributed {

class VocabParallelLogitsGather : public graph::GraphOperator {
public:
    VocabParallelLogitsGather(Tensor output,
                              const Tensor &local_logits,
                              infinicclComm_t communicator);
    VocabParallelLogitsGather(Tensor output,
                              const Tensor &local_logits,
                              Tensor workspace,
                              infinicclComm_t communicator);
    ~VocabParallelLogitsGather();
    void run() const override;
    static void execute(Tensor output,
                        const Tensor &local_logits,
                        infinicclComm_t communicator);
    static void execute(Tensor output,
                        const Tensor &local_logits,
                        Tensor workspace,
                        infinicclComm_t communicator);

private:
    void *planned_meta_;
};

Tensor vocab_parallel_logits_gather(const Tensor &local_logits,
                                    size_t world_size,
                                    infinicclComm_t communicator);
size_t vocab_parallel_logits_gather_workspace_numel(const Tensor &local_logits,
                                                    size_t world_size);
void vocab_parallel_logits_gather_(Tensor output,
                                   const Tensor &local_logits,
                                   infinicclComm_t communicator);
void vocab_parallel_logits_gather_(Tensor output,
                                   const Tensor &local_logits,
                                   Tensor workspace,
                                   infinicclComm_t communicator);

} // namespace infinicore::op::distributed
