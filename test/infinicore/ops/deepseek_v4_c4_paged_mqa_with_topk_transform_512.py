import argparse

import infinicore
import torch
from infinicore.lib import _infinicore


FP8_MAX = 448.0
TOPK = 512


def _as_core(tensor):
    return infinicore.from_torch(tensor)._underlying


def _sync():
    infinicore.sync_stream()
    torch.cuda.synchronize()


def _indexer_page_bytes(page_size):
    return (128 + 4) * page_size


def _run_case(pages, seq_lens_2d, graph):
    torch.manual_seed(20260810 + pages + int(seq_lens_2d) * 100 + int(graph) * 1000)
    device = "cuda"
    page_size = 64
    batch, heads = 5, 32
    max_c4_seq_len = pages * page_size
    blocks = batch * pages

    q = (torch.randn(batch, heads, 128, device=device, dtype=torch.bfloat16) * 0.2).contiguous()
    weights = torch.randn(batch, heads, device=device, dtype=torch.bfloat16).contiguous()
    cache_raw = torch.zeros(blocks, _indexer_page_bytes(page_size), device=device, dtype=torch.uint8).contiguous()
    cache_values = (torch.randn(blocks * page_size, 128, device=device, dtype=torch.bfloat16) * 0.2).contiguous()
    cache_indices = torch.arange(blocks * page_size, device=device, dtype=torch.int32)
    _infinicore.deepseek_v4_store_indexer_raw_cache_kernel_(
        _as_core(cache_values),
        _as_core(cache_raw),
        _as_core(cache_indices),
        page_size,
    )

    seq_lens = torch.tensor([0, 1, 73, min(511, max_c4_seq_len), max_c4_seq_len], device=device, dtype=torch.int32)
    seq_lens_for_fused = seq_lens.view(batch, 1).contiguous() if seq_lens_2d else seq_lens
    page_table = torch.arange(blocks, device=device, dtype=torch.int32).reshape(batch, pages).contiguous()

    q_fp8 = torch.empty(batch, heads, 128, device=device, dtype=torch.float8_e4m3fn)
    q_scale = torch.empty(batch, heads, 1, device=device, dtype=torch.float32)
    fused_weights = torch.empty(batch, heads, device=device, dtype=torch.float32)
    _infinicore.deepseek_v4_c4_act_quant_fused_scale_kernel_(
        _as_core(q),
        _as_core(weights),
        _as_core(q_fp8),
        _as_core(q_scale),
        _as_core(fused_weights),
        0.375,
    )

    logits = torch.empty(batch, max_c4_seq_len, device=device, dtype=torch.float32)
    ref_indices = torch.empty(batch, TOPK, device=device, dtype=torch.int32)
    _infinicore.deepseek_v4_c4_paged_mqa_logits_(
        _as_core(q_fp8),
        _as_core(fused_weights),
        _as_core(cache_raw),
        _as_core(seq_lens_for_fused),
        _as_core(page_table),
        _as_core(logits),
        max_c4_seq_len,
        page_size,
        False,
    )
    _infinicore.deepseek_v4_topk_transform_512_kernel_(
        _as_core(logits),
        _as_core(seq_lens),
        _as_core(page_table),
        _as_core(ref_indices),
        page_size,
    )

    got_indices = torch.empty_like(ref_indices)
    if graph:
        infinicore.start_graph_recording()
    _infinicore.deepseek_v4_c4_paged_mqa_with_topk_transform_512_(
        _as_core(q_fp8),
        _as_core(fused_weights),
        _as_core(cache_raw),
        _as_core(seq_lens_for_fused),
        _as_core(page_table),
        _as_core(got_indices),
        max_c4_seq_len,
        page_size,
        False,
    )
    if graph:
        graph_obj = infinicore.stop_graph_recording()
        got_indices.fill_(-7)
        _sync()
        graph_obj.run()
    _sync()

    assert torch.equal(got_indices, ref_indices), (
        f"mismatch pages={pages} seq_lens_2d={seq_lens_2d} graph={graph} "
        f"diff={(got_indices != ref_indices).sum().item()}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--skip-graph", action="store_true")
    args = parser.parse_args()

    for pages in (4, 16):
        for seq_lens_2d in (False, True):
            _run_case(pages, seq_lens_2d, graph=False)
            if not args.skip_graph:
                _run_case(pages, seq_lens_2d, graph=True)
    print("deepseek_v4_c4_paged_mqa_with_topk_transform_512 ok")


if __name__ == "__main__":
    main()
