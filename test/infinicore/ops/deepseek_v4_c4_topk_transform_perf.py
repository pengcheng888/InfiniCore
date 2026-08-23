import argparse
import time

import infinicore
import torch


TOPK = 512


def _as_core(tensor):
    return infinicore.from_torch(tensor)


def _sync():
    infinicore.sync_stream()
    torch.cuda.synchronize()


@torch.no_grad()
def sglang_topk_transform_512_pytorch_vectorized(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: torch.Tensor | None = None,
) -> None:
    batch_size = scores.shape[0]
    max_seq_len = scores.shape[1]
    device = scores.device
    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    page_mask = page_size - 1
    sequential_indices = torch.arange(TOPK, device=device, dtype=torch.int32).unsqueeze(0)
    sequential_valid = sequential_indices < seq_lens.unsqueeze(1)
    negative_indices = torch.full_like(sequential_indices, -1)
    if max_seq_len <= TOPK:
        raw_indices = torch.where(sequential_valid, sequential_indices, negative_indices)
        valid_topk = sequential_valid
    else:
        positions = torch.arange(max_seq_len, device=device).unsqueeze(0)
        valid_mask = positions < seq_lens.unsqueeze(1)
        masked_scores = scores.masked_fill(~valid_mask, float("-inf"))
        _, raw_indices = torch.topk(masked_scores, k=TOPK, dim=1, largest=True, sorted=False)
        raw_indices = raw_indices.to(torch.int32)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
        gathered_scores = scores[batch_indices, raw_indices]
        valid_topk = gathered_scores != float("-inf")
        needs_sequential = (seq_lens <= TOPK).unsqueeze(1)
        raw_indices = torch.where(
            needs_sequential,
            torch.where(sequential_valid, sequential_indices, negative_indices),
            raw_indices,
        )
        valid_topk = torch.where(needs_sequential, sequential_valid, valid_topk)

    page_idx = raw_indices >> page_bits
    offset_in_page = raw_indices & page_mask
    page_idx_clamped = torch.clamp(page_idx, min=0)
    physical_pages = torch.gather(page_tables, dim=1, index=page_idx_clamped.long())
    page_indices = (physical_pages << page_bits) | offset_in_page
    page_indices = page_indices.to(torch.int32)
    page_indices = torch.where(valid_topk, page_indices, negative_indices)
    out_page_indices.copy_(page_indices)
    if out_raw_indices is not None:
        raw_indices = torch.where(valid_topk, raw_indices, negative_indices)
        out_raw_indices.copy_(raw_indices)


def _make_seq_lens(batch, max_seq_len, device):
    if max_seq_len <= TOPK:
        values = torch.randint(1, max_seq_len + 1, (batch,), device=device, dtype=torch.int32)
    else:
        values = torch.randint(1, max_seq_len + 1, (batch,), device=device, dtype=torch.int32)
        anchors = torch.tensor([max_seq_len, TOPK + 1, TOPK, 64, 1], device=device, dtype=torch.int32)
        values[: min(batch, anchors.numel())] = anchors[: min(batch, anchors.numel())]
    return values.contiguous()


def _make_case(batch, max_seq_len, page_size, device, seed):
    torch.manual_seed(seed)
    scores = torch.randn(batch, max_seq_len, device=device, dtype=torch.float32).contiguous()
    seq_lens = _make_seq_lens(batch, max_seq_len, device)
    pages = (max_seq_len + page_size - 1) // page_size
    page_table = torch.arange(batch * pages, device=device, dtype=torch.int32).reshape(batch, pages).contiguous()
    out_torch = torch.empty(batch, TOPK, device=device, dtype=torch.int32)
    out_native = torch.empty_like(out_torch)
    return scores, seq_lens, page_table, out_torch, out_native


def _assert_outputs_match(name, out_torch, out_native, max_seq_len):
    if max_seq_len <= TOPK:
        ok = torch.equal(out_torch, out_native)
    else:
        ok = torch.equal(torch.sort(out_torch, dim=1).values, torch.sort(out_native, dim=1).values)
    if not ok:
        mismatch = (out_torch != out_native).sum().item()
        sorted_mismatch = (torch.sort(out_torch, dim=1).values != torch.sort(out_native, dim=1).values).sum().item()
        raise AssertionError(f"{name}: topk outputs mismatch raw={mismatch} sorted={sorted_mismatch}")


def _time_ms(fn, repeats, warmup):
    for _ in range(warmup):
        fn()
    _sync()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    _sync()
    return (time.perf_counter() - start) * 1000.0 / repeats


def _bench_case(name, batch, max_seq_len, page_size, repeats, warmup, device, seed):
    scores, seq_lens, page_table, out_torch, out_native = _make_case(batch, max_seq_len, page_size, device, seed)
    scores_core = _as_core(scores)
    seq_lens_core = _as_core(seq_lens)
    page_table_core = _as_core(page_table)
    out_native_core = _as_core(out_native)

    def torch_fn():
        sglang_topk_transform_512_pytorch_vectorized(scores, seq_lens, page_table, out_torch, page_size)

    def native_fn():
        infinicore.deepseek_v4_topk_transform_512_kernel_(
            scores_core,
            seq_lens_core,
            page_table_core,
            out_native_core,
            page_size,
        )

    torch_fn()
    native_fn()
    _sync()
    _assert_outputs_match(name, out_torch, out_native, max_seq_len)

    torch_ms = _time_ms(torch_fn, repeats, warmup)
    native_ms = _time_ms(native_fn, repeats, warmup)
    speedup = torch_ms / native_ms if native_ms > 0 else float("inf")
    print(
        f"{name:24s} batch={batch:5d} seq={max_seq_len:5d} "
        f"torch_vec={torch_ms:9.4f} ms  native={native_ms:9.4f} ms  speedup={speedup:7.2f}x"
    )
    return torch_ms, native_ms


def _default_cases(include_long):
    cases = [
        ("decode_384_b1", 1, 384, 64, None),
        ("small_384_b16", 16, 384, 64, None),
        ("bulk_384_b256", 256, 384, 64, None),
        ("bulk_384_b1024", 1024, 384, 64, None),
    ]
    if include_long:
        cases.extend(
            [
                ("topk_640_b1", 1, 640, 64, 1),
                ("topk_1024_b1", 1, 1024, 64, 1),
            ]
        )
    return cases


def main():
    parser = argparse.ArgumentParser(description="Benchmark SGLang torch-vectorized C4 topk transform vs InfiniCore native Hygon kernel.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--include-long", action="store_true", help="Also run slower seq_len > 512 native-kernel cases.")
    parser.add_argument("--batch", type=int, default=None, help="Run one custom case with this batch size.")
    parser.add_argument("--seq-len", type=int, default=None, help="Run one custom case with this max sequence length.")
    parser.add_argument("--page-size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda"
    print("C4 topk_transform_512 perf: SGLang PyTorch vectorized vs InfiniCore native Hygon")
    print(f"repeats={args.repeats} warmup={args.warmup} page_size={args.page_size}")
    if args.batch is not None or args.seq_len is not None:
        if args.batch is None or args.seq_len is None:
            raise SystemExit("--batch and --seq-len must be provided together")
        _bench_case("custom", args.batch, args.seq_len, args.page_size, args.repeats, args.warmup, device, 123)
        return

    for idx, (name, batch, max_seq_len, page_size, case_repeats) in enumerate(_default_cases(args.include_long)):
        repeats = case_repeats if case_repeats is not None else args.repeats
        warmup = 0 if case_repeats is not None else args.warmup
        _bench_case(name, batch, max_seq_len, page_size, repeats, warmup, device, 100 + idx)


if __name__ == "__main__":
    main()
