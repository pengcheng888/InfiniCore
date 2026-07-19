import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
from framework import GenericTestRunner
from mha_varlen import OpTest as MhaVarlenOpTest


class OpTest(MhaVarlenOpTest):
    def __init__(self):
        super().__init__()
        self.name = "Qwen3MhaVarlen"

    def infinicore_operator(
        self,
        query,
        k_cache,
        v_cache,
        block_tables,
        cum_seqlens_q,
        cum_seqlens_k,
        scale=1.0,
        max_seqlen_q=0,
        max_seqlen_k=0,
    ):
        if block_tables is None:
            key = k_cache
            value = v_cache
        else:
            key = k_cache.permute([0, 2, 1, 3])
            value = v_cache.permute([0, 2, 1, 3])
        out = infinicore.qwen3_mha_varlen(
            query,
            key,
            value,
            cum_seqlens_q,
            cum_seqlens_k,
            block_tables,
            max_seqlen_q,
            max_seqlen_k,
            alibi_slopes=None,
            scale=scale,
        )
        infinicore.sync_stream()
        return out


def main():
    GenericTestRunner(OpTest).run_and_exit()


if __name__ == "__main__":
    main()
