import argparse

import infinicore
import torch
from sgl_kernel import top_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.parse_args()

    torch.manual_seed(5)
    batch, length, topk = 2, 4096, 2048
    score = torch.randn(batch, length, device="cuda", dtype=torch.float32)
    lengths = torch.tensor([4096, 3072], device="cuda", dtype=torch.int32)
    src_page_table = torch.arange(batch * length, device="cuda", dtype=torch.int32).reshape(batch, length)
    cu_seqlens_q = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
    topk_indices_offset = torch.tensor([10, 20], device="cuda", dtype=torch.int32)

    ref_topk = top_k.fast_topk_v2(score, lengths, topk, None)
    ref_page = top_k.fast_topk_transform_fused(score, lengths, src_page_table, cu_seqlens_q, topk, None)
    ref_ragged = top_k.fast_topk_transform_ragged_fused(score, lengths, topk_indices_offset, topk, None)
    torch.cuda.synchronize()

    out_topk = torch.empty_like(ref_topk)
    out_page = torch.empty_like(ref_page)
    out_ragged = torch.empty_like(ref_ragged)

    infinicore.deepseek_v4_fast_topk_(
        infinicore.from_torch(score),
        infinicore.from_torch(out_topk),
        infinicore.from_torch(lengths),
        None,
    )
    infinicore.deepseek_v4_fast_topk_transform_fused_(
        infinicore.from_torch(score),
        infinicore.from_torch(lengths),
        infinicore.from_torch(out_page),
        infinicore.from_torch(src_page_table),
        infinicore.from_torch(cu_seqlens_q),
        None,
    )
    infinicore.deepseek_v4_fast_topk_transform_ragged_fused_(
        infinicore.from_torch(score),
        infinicore.from_torch(lengths),
        infinicore.from_torch(out_ragged),
        infinicore.from_torch(topk_indices_offset),
        None,
    )
    infinicore.sync_stream()

    assert torch.equal(torch.sort(out_topk, dim=1).values, torch.sort(ref_topk, dim=1).values)
    assert torch.equal(torch.sort(out_page, dim=1).values, torch.sort(ref_page, dim=1).values)
    assert torch.equal(torch.sort(out_ragged, dim=1).values, torch.sort(ref_ragged, dim=1).values)
    assert torch.all(out_topk >= 0)
    assert torch.all(out_page >= 0)
    assert torch.all(out_ragged >= 0)
    print("DeepseekV4FastTopK: passed")


if __name__ == "__main__":
    main()
