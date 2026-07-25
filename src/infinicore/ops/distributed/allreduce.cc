#include "infinicore/ops/distributed/allreduce.hpp"
#include "infinicore/ops/deepseek_v4_dcu_custom_allreduce.hpp"
#include "../../utils.hpp"
#include "../../../infiniccl/infiniccl_impl.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace infinicore::op::distributed {

namespace {

bool deepseek_v4_allreduce_fastpath_enabled() {
    const char *value = std::getenv("INFINICORE_ALLREDUCE_FASTPATH");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    const std::string text(value);
    if (text == "deepseek_v4" || text == "dsv4" || text == "hygon_deepseek_v4" || text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON") {
        return true;
    }
    if (text == "off" || text == "0" || text == "false" || text == "FALSE") {
        return false;
    }
    throw std::runtime_error("INFINICORE_ALLREDUCE_FASTPATH must be off or deepseek_v4");
}

bool try_deepseek_v4_allreduce_fastpath(Tensor output, const Tensor &input, infinicclReduceOp_t op, infinicclComm_t communicator) {
    if (!deepseek_v4_allreduce_fastpath_enabled() || communicator == nullptr || op != INFINICCL_SUM) {
        return false;
    }
    return infinicore::op::deepseek_v4_dcu_custom_allreduce_(
        output,
        input,
        communicator->rank,
        communicator->world_size);
}

} // namespace

struct PlannedMeta {
    graph::GraphTensor output, input;
    infinicclReduceOp_t op;
    infinicclComm_t communicator;
};

AllReduce::AllReduce(Tensor output, const Tensor &input, infinicclReduceOp_t op, infinicclComm_t communicator) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    INFINICORE_ASSERT(output->is_contiguous() && input->is_contiguous());
    INFINICORE_ASSERT(output->numel() == input->numel());
    planned_meta_ = new PlannedMeta{graph::GraphTensor(output), graph::GraphTensor(input), op, communicator};
}
AllReduce::~AllReduce() {
    if (planned_meta_) {
        PlannedMeta *meta = reinterpret_cast<PlannedMeta *>(planned_meta_);
        delete meta;
    }
}

void AllReduce::run() const {
    PlannedMeta *meta = reinterpret_cast<PlannedMeta *>(planned_meta_);

    if (try_deepseek_v4_allreduce_fastpath(meta->output, meta->input, meta->op, meta->communicator)) {
        return;
    }

    INFINICORE_CHECK_ERROR(infinicclAllReduce(meta->input->data(),
                                              meta->output->data(),
                                              meta->input->numel(),
                                              static_cast<infiniDtype_t>(static_cast<int>(meta->input->dtype())),
                                              meta->op,
                                              meta->communicator,
                                              infinicore::context::getStream()));
}

void AllReduce::execute(Tensor output, const Tensor &input, infinicclReduceOp_t op, infinicclComm_t communicator) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(AllReduce, output, input, op, communicator);
}

Tensor allreduce(const Tensor &input, infinicclReduceOp_t op, infinicclComm_t communicator) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    allreduce_(output, input, op, communicator);
    return output;
}

void allreduce_(Tensor output, const Tensor &input, infinicclReduceOp_t op, infinicclComm_t communicator) {
    AllReduce::execute(output, input, op, communicator);
}
} // namespace infinicore::op::distributed
