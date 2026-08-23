import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import sgl_kernel
import torch
from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorInitializer,
    TensorSpec,
    TestCase,
)


def parse_test_cases():
    return [
        TestCase(
            inputs=[
                TensorSpec.from_tensor(
                    (16, 8),
                    None,
                    infinicore.int32,
                    init_mode=TensorInitializer.RANDINT,
                    low=0,
                    high=64,
                ),
            ],
            kwargs={"num_experts": 64, "block_size": 16, "pad_sorted_token_ids": True},
            output_count=4,
            tolerance={"atol": 0, "rtol": 0},
            description="DeepseekV4MoeAlignBlockSize_int32",
        )
    ]


def _alloc_outputs_torch(topk_ids, num_experts, block_size, pad_sorted_token_ids):
    op_num_experts = num_experts + 1
    max_num_tokens_padded = topk_ids.numel() + op_num_experts * (block_size - 1)
    if pad_sorted_token_ids:
        max_num_tokens_padded = math.ceil(max_num_tokens_padded / block_size) * block_size
    sorted_token_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    experts_ids = torch.empty((math.ceil(max_num_tokens_padded / block_size),), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    cumsum_buffer = torch.empty((op_num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
    return sorted_token_ids, experts_ids, num_tokens_post_pad, cumsum_buffer


def _alloc_outputs_infini(topk_ids, num_experts, block_size, pad_sorted_token_ids):
    op_num_experts = num_experts + 1
    max_num_tokens_padded = topk_ids.numel() + op_num_experts * (block_size - 1)
    if pad_sorted_token_ids:
        max_num_tokens_padded = math.ceil(max_num_tokens_padded / block_size) * block_size
    sorted_token_ids = infinicore.empty((max_num_tokens_padded,), dtype=infinicore.int32, device=topk_ids.device)
    experts_ids = infinicore.empty((math.ceil(max_num_tokens_padded / block_size),), dtype=infinicore.int32, device=topk_ids.device)
    num_tokens_post_pad = infinicore.empty((1,), dtype=infinicore.int32, device=topk_ids.device)
    cumsum_buffer = infinicore.empty((op_num_experts + 1,), dtype=infinicore.int32, device=topk_ids.device)
    return sorted_token_ids, experts_ids, num_tokens_post_pad, cumsum_buffer


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("DeepseekV4MoeAlignBlockSize")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, topk_ids, num_experts=64, block_size=16, pad_sorted_token_ids=True, **kwargs):
        outputs = _alloc_outputs_torch(topk_ids, num_experts, block_size, pad_sorted_token_ids)
        sgl_kernel.moe_align_block_size(topk_ids, num_experts + 1, block_size, *outputs, pad_sorted_token_ids)
        torch.cuda.synchronize()
        return outputs

    def infinicore_operator(self, topk_ids, num_experts=64, block_size=16, pad_sorted_token_ids=True, **kwargs):
        outputs = _alloc_outputs_infini(topk_ids, num_experts, block_size, pad_sorted_token_ids)
        out = infinicore.deepseek_v4_moe_align_block_size_(
            topk_ids,
            num_experts + 1,
            block_size,
            outputs[0],
            outputs[1],
            outputs[2],
            outputs[3],
            pad_sorted_token_ids,
        )
        infinicore.sync_stream()
        return out


def main():
    GenericTestRunner(OpTest).run_and_exit()


if __name__ == "__main__":
    main()
