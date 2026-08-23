import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import sgl_kernel.flash_mla
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
                    (4, 128),
                    None,
                    infinicore.int32,
                    init_mode=TensorInitializer.MANUAL,
                    set_tensor=torch.arange(4 * 128, dtype=torch.int32).reshape(4, 128),
                ),
                TensorSpec.from_tensor(
                    (3,),
                    None,
                    infinicore.int32,
                    init_mode=TensorInitializer.MANUAL,
                    set_tensor=torch.tensor([0, 1, 3], dtype=torch.int32),
                ),
                TensorSpec.from_tensor(
                    (3,),
                    None,
                    infinicore.int32,
                    init_mode=TensorInitializer.MANUAL,
                    set_tensor=torch.tensor([1, 65, 128], dtype=torch.int32),
                ),
            ],
            kwargs={"page_size": 64, "max_pages": 2},
            tolerance={"atol": 0, "rtol": 0},
            description="DeepseekV4CreateFlashMLAKVIndices_int32",
        )
    ]


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("DeepseekV4CreateFlashMLAKVIndices")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, req_to_token, req_pool_indices, page_kernel_lens, page_size=64, max_pages=2, **kwargs):
        kv_indices = torch.full((req_pool_indices.numel(), max_pages), -1, dtype=torch.int32, device=req_to_token.device)
        sgl_kernel.flash_mla.dcu_create_flashmla_kv_indices(
            req_to_token,
            req_pool_indices,
            page_kernel_lens,
            None,
            kv_indices,
            req_to_token.stride(0),
            max_pages,
            page_size,
        )
        torch.cuda.synchronize()
        return kv_indices

    def infinicore_operator(self, req_to_token, req_pool_indices, page_kernel_lens, page_size=64, max_pages=2, **kwargs):
        kv_indices = infinicore.from_torch(
            torch.full(
                (req_pool_indices.numel(), max_pages),
                -1,
                dtype=torch.int32,
                device="cuda",
            )
        )
        out = infinicore.deepseek_v4_create_flashmla_kv_indices_(
            req_to_token,
            req_pool_indices,
            page_kernel_lens,
            None,
            kv_indices,
            req_to_token.stride(0),
            max_pages,
            page_size,
        )
        infinicore.sync_stream()
        return out


def main():
    GenericTestRunner(OpTest).run_and_exit()


if __name__ == "__main__":
    main()
