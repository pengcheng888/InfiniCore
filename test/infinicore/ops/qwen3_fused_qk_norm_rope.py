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
                TensorSpec.from_tensor((4, 4096), None, infinicore.bfloat16),
                TensorSpec.from_tensor((128,), None, infinicore.bfloat16),
                TensorSpec.from_tensor((128,), None, infinicore.bfloat16),
                TensorSpec.from_tensor(
                    (4,),
                    None,
                    infinicore.int32,
                    init_mode=TensorInitializer.RANDINT,
                    low=0,
                    high=32,
                ),
            ],
            kwargs={
                "num_heads_q": 16,
                "num_heads_k": 8,
                "num_heads_v": 8,
                "head_dim": 128,
                "eps": 1e-6,
                "base": 1000000.0,
                "is_neox": True,
                "factor": 1.0,
                "low": 0.0,
                "high": 0.0,
                "attention_factor": 1.0,
                "rotary_dim": 128,
            },
            comparison_target=0,
            tolerance={"atol": 2e-2, "rtol": 2e-2},
            description="Qwen3FusedQKNormRoPE_bfloat16",
        )
    ]


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("Qwen3FusedQKNormRoPE")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, qkv, q_weight, k_weight, position_ids, **kwargs):
        sgl_kernel.fused_qk_norm_rope(
            qkv,
            kwargs["num_heads_q"],
            kwargs["num_heads_k"],
            kwargs["num_heads_v"],
            kwargs["head_dim"],
            kwargs["eps"],
            q_weight,
            k_weight,
            kwargs["base"],
            kwargs["is_neox"],
            position_ids,
            kwargs["factor"],
            kwargs["low"],
            kwargs["high"],
            kwargs["attention_factor"],
            kwargs["rotary_dim"],
        )
        torch.cuda.synchronize()
        return qkv

    def infinicore_operator(self, qkv, q_weight, k_weight, position_ids, **kwargs):
        out = infinicore.qwen3_fused_qk_norm_rope_(
            qkv,
            kwargs["num_heads_q"],
            kwargs["num_heads_k"],
            kwargs["num_heads_v"],
            kwargs["head_dim"],
            kwargs["eps"],
            q_weight,
            k_weight,
            kwargs["base"],
            kwargs["is_neox"],
            position_ids,
            kwargs["factor"],
            kwargs["low"],
            kwargs["high"],
            kwargs["attention_factor"],
            kwargs["rotary_dim"],
        )
        infinicore.sync_stream()
        return out


def main():
    GenericTestRunner(OpTest).run_and_exit()


if __name__ == "__main__":
    main()

