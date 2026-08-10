import argparse
import statistics
import time

import infinicore
import torch
from infinicore.lib import _infinicore


# Defaults mirror DeepSeek-V4 config.json:
# max_position_embeddings=1048576, index_topk=512, compress_ratios include C4=4.
DSV4_MAX_POSITION_EMBEDDINGS = 1048576
DSV4_C4_COMPRESS_RATIO = 4
DSV4_INDEX_TOPK = 512
DSV4_C4_PAGE_SIZE = 64
DSV4_MAX_C4_SEQ_LEN = DSV4_MAX_POSITION_EMBEDDINGS // DSV4_C4_COMPRESS_RATIO
TOPK = DSV4_INDEX_TOPK
DEFAULT_TOKENS = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"
DEFAULT_SEQ_LENS = "384"


def _parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _as_core(tensor):
    return infinicore.from_torch(tensor)._underlying


def _sync():
    infinicore.sync_stream()
    torch.cuda.synchronize()


@torch.no_grad()
def _torch_topk_transform_512(scores, seq_lens, page_table, out_page_indices, page_size):
    batch, max_seq_len = scores.shape
    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    page_mask = page_size - 1
    seq_lens = torch.clamp(seq_lens, min=0, max=max_seq_len)

    sequential_indices = torch.arange(TOPK, device=scores.device, dtype=torch.int32).unsqueeze(0)
    sequential_valid = sequential_indices < seq_lens.unsqueeze(1)
    negative_indices = torch.full_like(sequential_indices, -1)
    if max_seq_len <= TOPK:
        raw_indices = torch.where(sequential_valid, sequential_indices, negative_indices)
        valid_topk = sequential_valid
    else:
        positions = torch.arange(max_seq_len, device=scores.device).unsqueeze(0)
        valid_mask = positions < seq_lens.unsqueeze(1)
        masked_scores = scores.masked_fill(~valid_mask, float("-inf"))
        _, raw_indices = torch.topk(masked_scores, k=TOPK, dim=1, largest=True, sorted=False)
        raw_indices = raw_indices.to(torch.int32)
        needs_sequential = seq_lens.unsqueeze(1) <= TOPK
        raw_indices = torch.where(
            needs_sequential,
            torch.where(sequential_valid, sequential_indices, negative_indices),
            raw_indices,
        )
        valid_topk = torch.where(needs_sequential, sequential_valid, torch.ones_like(sequential_valid))

    page_idx = raw_indices >> page_bits
    offset = raw_indices & page_mask
    physical_pages = torch.gather(page_table, dim=1, index=torch.clamp(page_idx, min=0).long())
    page_indices = ((physical_pages << page_bits) | offset).to(torch.int32)
    out_page_indices.copy_(torch.where(valid_topk, page_indices, negative_indices))


def _make_seq_lens(batch, max_seq_len, device):
    if max_seq_len <= TOPK:
        values = torch.randint(0, max_seq_len + 1, (batch,), device=device, dtype=torch.int32)
    else:
        values = torch.randint(1, max_seq_len + 1, (batch,), device=device, dtype=torch.int32)
        anchors = torch.tensor([0, 1, 64, TOPK, TOPK + 1, max_seq_len], device=device, dtype=torch.int32)
        values[: min(batch, anchors.numel())] = anchors[: min(batch, anchors.numel())]
    return values.contiguous()


def _make_case(batch, max_seq_len, page_size, seed):
    torch.manual_seed(seed)
    device = "cuda"
    scores = torch.randn(batch, max_seq_len, device=device, dtype=torch.float32).contiguous()
    seq_lens = _make_seq_lens(batch, max_seq_len, device)
    pages = (max_seq_len + page_size - 1) // page_size
    page_table = torch.arange(batch * pages, device=device, dtype=torch.int32).reshape(batch, pages).contiguous()
    return scores, seq_lens, page_table


def _assert_match(name, ref, got, max_seq_len):
    if max_seq_len <= TOPK:
        ok = torch.equal(ref, got)
    else:
        ok = torch.equal(torch.sort(ref, dim=1).values, torch.sort(got, dim=1).values)
    if not ok:
        raw_diff = (ref != got).sum().item()
        sorted_diff = (torch.sort(ref, dim=1).values != torch.sort(got, dim=1).values).sum().item()
        raise AssertionError(f"{name}: mismatch raw_diff={raw_diff} sorted_diff={sorted_diff}")


def _measure_ms(fn, repeats, warmup):
    for _ in range(warmup):
        fn()
    _sync()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.mean(samples), statistics.median(samples)


def _bench_case(name, tokens, max_seq_len, page_size, repeats, warmup, check, seed):
    scores, seq_lens, page_table = _make_case(tokens, max_seq_len, page_size, seed)
    out_torch = torch.empty(tokens, TOPK, device="cuda", dtype=torch.int32)
    out_old = torch.empty_like(out_torch)
    out_sglang = torch.empty_like(out_torch)

    scores_core = _as_core(scores)
    seq_lens_core = _as_core(seq_lens)
    page_table_core = _as_core(page_table)
    out_old_core = _as_core(out_old)
    out_sglang_core = _as_core(out_sglang)

    def torch_fn():
        _torch_topk_transform_512(scores, seq_lens, page_table, out_torch, page_size)

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

    if check:
        torch_fn()
        old_fn()
        sglang_fn()
        _sync()
        _assert_match(name + "/old", out_torch, out_old, max_seq_len)
        _assert_match(name + "/sglang", out_torch, out_sglang, max_seq_len)

    torch_avg, torch_med = _measure_ms(torch_fn, repeats, warmup)
    old_avg, old_med = _measure_ms(old_fn, repeats, warmup)
    sglang_avg, sglang_med = _measure_ms(sglang_fn, repeats, warmup)
    old_speedup = old_avg / sglang_avg if sglang_avg > 0 else float("inf")
    torch_speedup = torch_avg / sglang_avg if sglang_avg > 0 else float("inf")

    print(
        f"{name:22s} tokens={tokens:5d} seq={max_seq_len:6d} page={page_size:3d} "
        f"torch_avg={torch_avg:9.4f} torch_med={torch_med:9.4f} "
        f"old_avg={old_avg:9.4f} old_med={old_med:9.4f} "
        f"sgl_avg={sglang_avg:9.4f} sgl_med={sglang_med:9.4f} "
        f"old/sgl={old_speedup:7.2f}x torch/sgl={torch_speedup:7.2f}x"
    )


def _validate_seq_lens(seq_lens):
    for seq_len in seq_lens:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if seq_len > DSV4_MAX_C4_SEQ_LEN:
            raise ValueError(
                f"seq_len={seq_len} exceeds DeepSeek-V4 C4 max sequence length "
                f"{DSV4_MAX_C4_SEQ_LEN} (= max_position_embeddings/compress_ratio_c4)"
            )


def main():
    parser = argparse.ArgumentParser(description="Benchmark deepseek_v4_topk_transform_512_sglang_kernel_.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--tokens", default=DEFAULT_TOKENS, help="Comma-separated token counts to sweep.")
    parser.add_argument("--seq-lens", default=DEFAULT_SEQ_LENS, help="Comma-separated C4 score lengths to sweep.")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--page-size", type=int, default=DSV4_C4_PAGE_SIZE)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--check", dest="check", action="store_true", default=True)
    parser.add_argument("--no-check", dest="check", action="store_false")
    args = parser.parse_args()

    print("deepseek_v4_topk_transform_512_sglang perf")
    print(
        "config: "
        f"index_topk={DSV4_INDEX_TOPK} c4_page_size={DSV4_C4_PAGE_SIZE} "
        f"max_position_embeddings={DSV4_MAX_POSITION_EMBEDDINGS} "
        f"c4_compress_ratio={DSV4_C4_COMPRESS_RATIO} max_c4_seq_len={DSV4_MAX_C4_SEQ_LEN}"
    )
    print(
        f"repeats={args.repeats} warmup={args.warmup} page_size={args.page_size} "
        f"tokens={args.tokens} seq_lens={args.seq_lens} check={args.check}"
    )

    if args.batch is not None or args.seq_len is not None:
        if args.batch is None or args.seq_len is None:
            raise SystemExit("--batch and --seq-len must be provided together")
        _bench_case("custom", args.batch, args.seq_len, args.page_size, args.repeats, args.warmup, args.check, 20260810)
        return

    tokens_list = _parse_int_list(args.tokens)
    seq_lens = _parse_int_list(args.seq_lens)
    _validate_seq_lens(seq_lens)

    case_idx = 0
    for max_seq_len in seq_lens:
        for tokens in tokens_list:
            name = f"tok{tokens}_seq{max_seq_len}"
            _bench_case(name, tokens, max_seq_len, args.page_size, args.repeats, args.warmup, args.check, 20260810 + case_idx)
            case_idx += 1


if __name__ == "__main__":
    main()
