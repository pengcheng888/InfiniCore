import argparse
import time

import infinicore
import torch
from infinicore.lib import _infinicore


TOPK = 512


def _as_core(tensor):
    return infinicore.from_torch(tensor)._underlying


def _sync():
    infinicore.sync_stream()
    torch.cuda.synchronize()


@torch.no_grad()
def _torch_ref(scores, seq_lens, page_table, out, page_size):
    batch, max_seq_len = scores.shape
    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    page_mask = page_size - 1
    seq_lens_clamped = torch.clamp(seq_lens, min=0, max=max_seq_len)

    sequential_indices = torch.arange(TOPK, device=scores.device, dtype=torch.int32).unsqueeze(0)
    sequential_valid = sequential_indices < seq_lens_clamped.unsqueeze(1)
    negative_indices = torch.full_like(sequential_indices, -1)

    if max_seq_len <= TOPK:
        raw_indices = torch.where(sequential_valid, sequential_indices, negative_indices)
        valid_topk = sequential_valid
    else:
        positions = torch.arange(max_seq_len, device=scores.device).unsqueeze(0)
        valid_mask = positions < seq_lens_clamped.unsqueeze(1)
        masked_scores = scores.masked_fill(~valid_mask, float("-inf"))
        _, raw_indices = torch.topk(masked_scores, k=TOPK, dim=1, largest=True, sorted=False)
        raw_indices = raw_indices.to(torch.int32)
        valid_topk = seq_lens_clamped.unsqueeze(1) > TOPK
        needs_sequential = seq_lens_clamped.unsqueeze(1) <= TOPK
        raw_indices = torch.where(
            needs_sequential,
            torch.where(sequential_valid, sequential_indices, negative_indices),
            raw_indices,
        )
        valid_topk = torch.where(needs_sequential, sequential_valid, valid_topk)

    page_idx = raw_indices >> page_bits
    offset = raw_indices & page_mask
    physical_pages = torch.gather(page_table, dim=1, index=torch.clamp(page_idx, min=0).long())
    page_indices = ((physical_pages << page_bits) | offset).to(torch.int32)
    out.copy_(torch.where(valid_topk, page_indices, negative_indices))


def _make_case(batch, max_seq_len, page_size, seed):
    torch.manual_seed(seed)
    device = "cuda"
    scores = torch.randn(batch, max_seq_len, device=device, dtype=torch.float32).contiguous()
    if max_seq_len <= TOPK:
        seq_lens = torch.randint(0, max_seq_len + 1, (batch,), device=device, dtype=torch.int32)
    else:
        seq_lens = torch.randint(1, max_seq_len + 1, (batch,), device=device, dtype=torch.int32)
        anchors = torch.tensor([0, 1, 64, TOPK, TOPK + 1, max_seq_len], device=device, dtype=torch.int32)
        seq_lens[: min(batch, anchors.numel())] = anchors[: min(batch, anchors.numel())]
    seq_lens = seq_lens.contiguous()

    pages = (max_seq_len + page_size - 1) // page_size
    page_table = torch.arange(batch * pages, device=device, dtype=torch.int32).reshape(batch, pages).contiguous()
    return scores, seq_lens, page_table


def _assert_match(name, scores, seq_lens, ref, got, max_seq_len, page_size):
    del scores, seq_lens, page_size
    if max_seq_len <= TOPK:
        ok = torch.equal(ref, got)
    else:
        ok = torch.equal(torch.sort(ref, dim=1).values, torch.sort(got, dim=1).values)
    if not ok:
        raw_diff = (ref != got).sum().item()
        sorted_diff = (torch.sort(ref, dim=1).values != torch.sort(got, dim=1).values).sum().item()
        raise AssertionError(f"{name}: mismatch raw_diff={raw_diff} sorted_diff={sorted_diff}")


def _time_ms(fn, repeats, warmup):
    for _ in range(warmup):
        fn()
    _sync()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    _sync()
    return (time.perf_counter() - start) * 1000.0 / repeats


def _run_case(name, batch, max_seq_len, page_size, repeats, warmup, seed, graph):
    scores, seq_lens, page_table = _make_case(batch, max_seq_len, page_size, seed)
    out_ref = torch.empty(batch, TOPK, device="cuda", dtype=torch.int32)
    out_old = torch.empty_like(out_ref)
    out_sglang = torch.empty_like(out_ref)

    scores_core = _as_core(scores)
    seq_lens_core = _as_core(seq_lens)
    page_table_core = _as_core(page_table)
    out_old_core = _as_core(out_old)
    out_sglang_core = _as_core(out_sglang)

    def torch_fn():
        _torch_ref(scores, seq_lens, page_table, out_ref, page_size)

    def old_fn():
        _infinicore.deepseek_v4_topk_transform_512_kernel_(
            scores_core,
            seq_lens_core,
            page_table_core,
            out_old_core,
            page_size,
        )

    def sglang_fn():
        _infinicore.deepseek_v4_topk_transform_512_sglang_kernel_(
            scores_core,
            seq_lens_core,
            page_table_core,
            out_sglang_core,
            page_size,
        )

    torch_fn()
    old_fn()
    sglang_fn()
    _sync()
    _assert_match(name + "/old", scores, seq_lens, out_ref, out_old, max_seq_len, page_size)
    _assert_match(name + "/sglang", scores, seq_lens, out_ref, out_sglang, max_seq_len, page_size)

    if graph:
        out_sglang.fill_(-7)
        _sync()
        infinicore.start_graph_recording()
        sglang_fn()
        graph_obj = infinicore.stop_graph_recording()
        out_sglang.fill_(-7)
        _sync()
        graph_obj.run()
        _sync()
        _assert_match(name + "/sglang_graph", scores, seq_lens, out_ref, out_sglang, max_seq_len, page_size)

    old_ms = _time_ms(old_fn, repeats, warmup)
    sglang_ms = _time_ms(sglang_fn, repeats, warmup)
    speedup = old_ms / sglang_ms if sglang_ms > 0 else float("inf")
    print(
        f"{name:18s} batch={batch:5d} seq={max_seq_len:6d} page={page_size:3d} "
        f"old={old_ms:9.4f} ms sglang={sglang_ms:9.4f} ms speedup={speedup:7.2f}x"
    )


def _default_cases():
    return [
        ("short_b1", 1, 384, 64),
        ("short_b64", 64, 384, 64),
        ("topk_b1", 1, 1024, 64),
        ("topk_b16", 16, 4096, 64),
    ]


def main():
    parser = argparse.ArgumentParser(description="Check and benchmark InfiniCore SGLang-style topk_transform_512 kernel.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--skip-graph", action="store_true")
    args = parser.parse_args()

    if args.batch is not None or args.seq_len is not None:
        if args.batch is None or args.seq_len is None:
            raise SystemExit("--batch and --seq-len must be provided together")
        _run_case("custom", args.batch, args.seq_len, args.page_size, args.repeats, args.warmup, 20260810, not args.skip_graph)
        return

    for idx, (name, batch, max_seq_len, page_size) in enumerate(_default_cases()):
        _run_case(name, batch, max_seq_len, page_size, args.repeats, args.warmup, 20260810 + idx, not args.skip_graph)

    print("deepseek_v4_topk_transform_512_sglang ok")


if __name__ == "__main__":
    main()
