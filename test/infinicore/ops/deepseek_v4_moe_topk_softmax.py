import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import sgl_kernel
import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase


def parse_test_cases():
    return [
        TestCase(
            inputs=[
                TensorSpec.from_tensor((16, 64), None, infinicore.float32),
            ],
            kwargs={"topk": 8, "renormalize": True, "moe_softcapping": 0.0},
            output_count=2,
            tolerance={"atol": 1e-5, "rtol": 1e-5},
            description="DeepseekV4MoeTopkSoftmax_fp32",
        )
    ]


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("DeepseekV4MoeTopkSoftmax")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, gating_output, topk=8, renormalize=True, moe_softcapping=0.0, **kwargs):
        values = torch.empty((gating_output.shape[0], topk), dtype=torch.float32, device=gating_output.device)
        indices = torch.empty((gating_output.shape[0], topk), dtype=torch.int32, device=gating_output.device)
        sgl_kernel.topk_softmax(values, indices, gating_output, renormalize, moe_softcapping, None)
        torch.cuda.synchronize()
        return values, indices

    def infinicore_operator(self, gating_output, topk=8, renormalize=True, moe_softcapping=0.0, **kwargs):
        values = infinicore.empty((gating_output.shape[0], topk), dtype=infinicore.float32, device=gating_output.device)
        indices = infinicore.empty((gating_output.shape[0], topk), dtype=infinicore.int32, device=gating_output.device)
        out = infinicore.deepseek_v4_moe_topk_softmax_(values, indices, gating_output, renormalize, moe_softcapping, None)
        infinicore.sync_stream()
        return out


def main():
    GenericTestRunner(OpTest).run_and_exit()


if __name__ == "__main__":
    main()
