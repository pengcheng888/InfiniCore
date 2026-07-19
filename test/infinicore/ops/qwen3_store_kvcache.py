import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorInitializer,
    TensorSpec,
    TestCase,
)


def parse_test_cases():
    cases = []
    for dtype in [infinicore.float16, infinicore.bfloat16]:
        cases.append(
            TestCase(
                inputs=[
                    TensorSpec.from_tensor((4, 8, 128), None, dtype),
                    TensorSpec.from_tensor((4, 8, 128), None, dtype),
                    TensorSpec.from_tensor(
                        (16, 8, 128),
                        None,
                        dtype,
                        init_mode=TensorInitializer.ZEROS,
                    ),
                    TensorSpec.from_tensor(
                        (16, 8, 128),
                        None,
                        dtype,
                        init_mode=TensorInitializer.ZEROS,
                    ),
                    TensorSpec.from_tensor(
                        (4,),
                        None,
                        infinicore.int64,
                        init_mode=TensorInitializer.MANUAL,
                        set_tensor=torch.tensor([0, 3, 7, 11], dtype=torch.int64),
                    ),
                ],
                output_count=2,
                comparison_target=[2, 3],
                tolerance={"atol": 0.0, "rtol": 0.0},
                description=f"Qwen3StoreKVCache_{dtype}",
            )
        )
    return cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("Qwen3StoreKVCache")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, k, v, k_cache, v_cache, indices, **kwargs):
        k_cache.index_copy_(0, indices, k)
        v_cache.index_copy_(0, indices, v)
        torch.cuda.synchronize()
        return k_cache, v_cache

    def infinicore_operator(self, k, v, k_cache, v_cache, indices, **kwargs):
        out = infinicore.qwen3_store_kvcache_(k, v, k_cache, v_cache, indices)
        infinicore.sync_stream()
        return out


def main():
    GenericTestRunner(OpTest).run_and_exit()


if __name__ == "__main__":
    main()
