import argparse

import infinicore
import torch
from sgl_kernel import kvcacheio


def _as_core(tensor):
    return infinicore.from_torch(tensor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.parse_args()

    bs = 4
    page_size = 64
    free_pages = torch.arange(100, 1124, device="cuda", dtype=torch.int64)

    seq_lens_decode = torch.tensor([1, 2, 3, 4], device="cuda", dtype=torch.int64)
    last_loc_decode = torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.int32)
    ref_decode = torch.full((bs,), -1, device="cuda", dtype=torch.int64)
    kvcacheio.dcu_alloc_decode_kernel(seq_lens_decode, last_loc_decode, free_pages, ref_decode, bs, page_size)
    torch.cuda.synchronize()

    out_decode = torch.full_like(ref_decode, -1)
    infinicore.deepseek_v4_dcu_alloc_decode_kernel_(
        _as_core(seq_lens_decode),
        _as_core(last_loc_decode),
        _as_core(free_pages),
        _as_core(out_decode),
        bs,
        page_size,
    )
    infinicore.sync_stream()
    assert torch.equal(out_decode, ref_decode)

    pre_lens = torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.int64)
    seq_lens_extend = torch.tensor([3, 5, 7, 9], device="cuda", dtype=torch.int64)
    last_loc_extend = torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.int64)
    ref_extend = torch.full((bs * 8,), -1, device="cuda", dtype=torch.int64)
    kvcacheio.dcu_alloc_extend_kernel(
        pre_lens,
        seq_lens_extend,
        last_loc_extend,
        free_pages,
        ref_extend,
        bs,
        page_size,
    )
    torch.cuda.synchronize()

    out_extend = torch.full_like(ref_extend, -1)
    infinicore.deepseek_v4_dcu_alloc_extend_kernel_(
        _as_core(pre_lens),
        _as_core(seq_lens_extend),
        _as_core(last_loc_extend),
        _as_core(free_pages),
        _as_core(out_extend),
        bs,
        page_size,
    )
    infinicore.sync_stream()
    assert torch.equal(out_extend, ref_extend)

    print("DeepseekV4DcuCacheAlloc: passed")


if __name__ == "__main__":
    main()
