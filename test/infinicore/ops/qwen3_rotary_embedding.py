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


_DTYPES = [infinicore.bfloat16, infinicore.float16]


def parse_test_cases():
    cases = []
    for dtype in _DTYPES:
        cases.append(
            TestCase(
                inputs=[
                    TensorSpec.from_tensor(
                        (4,),
                        None,
                        infinicore.int64,
                        init_mode=TensorInitializer.RANDINT,
                        low=0,
                        high=32,
                    ),
                    TensorSpec.from_tensor((4, 16, 128), None, dtype),
                    TensorSpec.from_tensor((4, 8, 128), None, dtype),
                    TensorSpec.from_tensor((32, 128), None, dtype),
                ],
                kwargs={"head_size": 128, "is_neox": True},
                comparison_target=[1, 2],
                output_count=2,
                tolerance={"atol": 1e-2, "rtol": 1e-2},
                description=f"Qwen3RotaryEmbedding_{dtype}",
            )
        )
    return cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("Qwen3RotaryEmbedding")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, positions, query, key, cos_sin_cache, **kwargs):
        sgl_kernel.rotary_embedding(
            positions,
            query,
            key,
            kwargs["head_size"],
            cos_sin_cache,
            kwargs["is_neox"],
        )
        torch.cuda.synchronize()
        return query, key

    def infinicore_operator(self, positions, query, key, cos_sin_cache, **kwargs):
        out = infinicore.qwen3_rotary_embedding_(
            positions,
            query,
            key,
            kwargs["head_size"],
            cos_sin_cache,
            kwargs["is_neox"],
        )
        infinicore.sync_stream()
        return out


def main():
    GenericTestRunner(OpTest).run_and_exit()


if __name__ == "__main__":
    main()
