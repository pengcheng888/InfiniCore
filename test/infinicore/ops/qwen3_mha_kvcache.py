import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
from framework import GenericTestRunner
from mha_kvcache import OpTest as MhaKVCacheOpTest


class OpTest(MhaKVCacheOpTest):
    def __init__(self):
        super().__init__()
        self.name = "Qwen3MhaKVCache"

    def infinicore_operator(
        self,
        q,
        k_cache,
        v_cache,
        seqlens_k,
        block_table,
        scale=1.0,
    ):
        out = infinicore.qwen3_mha_kvcache(
            q,
            k_cache,
            v_cache,
            seqlens_k,
            block_table,
            alibi_slopes=None,
            scale=scale,
        )
        infinicore.sync_stream()
        return out


def main():
    GenericTestRunner(OpTest).run_and_exit()


if __name__ == "__main__":
    main()
