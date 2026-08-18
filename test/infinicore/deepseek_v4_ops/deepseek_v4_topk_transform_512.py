import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
from infinicore.lib import _infinicore


DSV4_MAX_POSITION_EMBEDDINGS = 1048576
DSV4_C4_COMPRESS_RATIO = 4
DSV4_INDEX_TOPK = 512
DSV4_C4_PAGE_SIZE = 64
DSV4_MAX_C4_SEQ_LEN = DSV4_MAX_POSITION_EMBEDDINGS // DSV4_C4_COMPRESS_RATIO
TOPK = DSV4_INDEX_TOPK
DEFAULT_TOKENS = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"
DEFAULT_SEQ_LENS = "384"


def _parse_tokens(text):
    return [int(item) for item in text.split(",") if item.strip()]


def _as_core(tensor):
    return infinicore.from_torch(tensor)


@torch.no_grad()
def _torch_ref(scores, seq_lens, page_table, out_page_indices, page_size):
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
    return out_page_indices


def _make_seq_lens(batch, max_seq_len, device):
    if max_seq_len <= TOPK:
        values = torch.randint(0, max_seq_len + 1, (batch,), device=device, dtype=torch.int32)
    else:
        values = torch.randint(1, max_seq_len + 1, (batch,), device=device, dtype=torch.int32)
        anchors = torch.tensor([0, 1, 64, TOPK, TOPK + 1, max_seq_len], device=device, dtype=torch.int32)
        values[: min(batch, anchors.numel())] = anchors[: min(batch, anchors.numel())]
    return values.contiguous()


def _make_inputs(batch, max_seq_len, page_size, seed):
    torch.manual_seed(seed + batch * 17 + max_seq_len)
    device = torch.device("cuda")
    scores = torch.randn(batch, max_seq_len, device=device, dtype=torch.float32).contiguous()
    seq_lens = _make_seq_lens(batch, max_seq_len, device)
    pages = (max_seq_len + page_size - 1) // page_size
    page_table = torch.arange(batch * pages, device=device, dtype=torch.int32).reshape(batch, pages).contiguous()
    return scores, seq_lens, page_table


def _compare(lhs, rhs, max_seq_len):
    if max_seq_len <= TOPK:
        lhs_cmp = lhs
        rhs_cmp = rhs
    else:
        lhs_cmp = torch.sort(lhs, dim=1).values
        rhs_cmp = torch.sort(rhs, dim=1).values
    diff = (lhs_cmp.int() - rhs_cmp.int()).abs()
    max_abs = diff.max().item() if diff.numel() > 0 else 0
    allclose = torch.equal(lhs_cmp, rhs_cmp)
    return max_abs, 0.0, allclose


def _bench(fn, warmup, iters):
    warmup_value = None
    for _ in range(warmup):
        warmup_value = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    return {
        "avg_ms": total_ms / iters,
        "total_ms": total_ms,
        "warmup_value": warmup_value,
    }


def _run_case(tokens, max_seq_len, args, case_idx):
    scores, seq_lens, page_table = _make_inputs(tokens, max_seq_len, args.page_size, args.seed + case_idx)
    ref_out = torch.empty(tokens, TOPK, device=scores.device, dtype=torch.int32)
    kernel_out = torch.empty_like(ref_out)
    sglang_out = torch.empty_like(ref_out)

    core_scores = _as_core(scores)
    core_seq_lens = _as_core(seq_lens)
    core_page_table = _as_core(page_table)
    core_kernel_out = _as_core(kernel_out)
    core_sglang_out = _as_core(sglang_out)

    def run_ref():
        return _torch_ref(scores, seq_lens, page_table, ref_out, args.page_size)

    def run_kernel():
        _infinicore.deepseek_v4_topk_transform_512_kernel_(
            core_scores._underlying,
            core_seq_lens._underlying,
            core_page_table._underlying,
            core_kernel_out._underlying,
            args.page_size,
        )
        return kernel_out

    def run_sglang():
        _infinicore.deepseek_v4_topk_transform_512_sglang_kernel_(
            core_scores._underlying,
            core_seq_lens._underlying,
            core_page_table._underlying,
            core_sglang_out._underlying,
            args.page_size,
        )
        return sglang_out

    ref_perf = _bench(run_ref, args.warmup, args.iters)
    kernel_perf = _bench(run_kernel, args.warmup, args.iters)
    sglang_perf = _bench(run_sglang, args.warmup, args.iters)

    ref = ref_perf["warmup_value"]
    kernel_max_abs, kernel_max_rel, kernel_allclose = _compare(kernel_perf["warmup_value"], ref, max_seq_len)
    sglang_max_abs, sglang_max_rel, sglang_allclose = _compare(sglang_perf["warmup_value"], ref, max_seq_len)

    return {
        "tokens": tokens,
        "seq_len": max_seq_len,
        "ref_avg": ref_perf["avg_ms"],
        "kernel_avg": kernel_perf["avg_ms"],
        "sglang_avg": sglang_perf["avg_ms"],
        "kernel_speedup": ref_perf["avg_ms"] / kernel_perf["avg_ms"] if kernel_perf["avg_ms"] > 0 else float("inf"),
        "sglang_speedup": ref_perf["avg_ms"] / sglang_perf["avg_ms"] if sglang_perf["avg_ms"] > 0 else float("inf"),
        "max_abs": kernel_max_abs,
        "max_rel": kernel_max_rel,
        "allclose": bool(kernel_allclose),
        "sglang_max_abs": sglang_max_abs,
        "sglang_max_rel": sglang_max_rel,
        "sglang_allclose": bool(sglang_allclose),
    }


def _validate_seq_lens(seq_lens):
    for seq_len in seq_lens:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if seq_len > DSV4_MAX_C4_SEQ_LEN:
            raise ValueError(
                f"seq_len={seq_len} exceeds DeepSeek-V4 C4 max sequence length "
                f"{DSV4_MAX_C4_SEQ_LEN} (= max_position_embeddings/compress_ratio_c4)"
            )


def _print_header(args):
    print(
        "config: "
        f"index_topk={DSV4_INDEX_TOPK} c4_page_size={DSV4_C4_PAGE_SIZE} "
        f"max_c4_seq_len={DSV4_MAX_C4_SEQ_LEN} page_size={args.page_size}"
    )
    print(
        f"{'tokens':>8} | {'seq_len':>7} | {'ref avg':>10} | {'kernel avg':>10} | "
        f"{'kernel spd':>10} | {'sglang avg':>10} | {'sgl spd':>8} | "
        f"{'max_abs':>8} | {'allclose':>8} | {'sgl_abs':>8} | {'sgl_ok':>6}"
    )
    print("-" * 128)


def _print_row(result):
    print(
        f"{result['tokens']:8d} | "
        f"{result['seq_len']:7d} | "
        f"{result['ref_avg']:10.4f} | "
        f"{result['kernel_avg']:10.4f} | "
        f"{result['kernel_speedup']:10.2f} | "
        f"{result['sglang_avg']:10.4f} | "
        f"{result['sglang_speedup']:8.2f} | "
        f"{result['max_abs']:8d} | "
        f"{str(result['allclose']):>8} | "
        f"{result['sglang_max_abs']:8d} | "
        f"{str(result['sglang_allclose']):>6}"
    )


def main():
    parser = argparse.ArgumentParser(description="Check and benchmark DeepSeek-V4 topk_transform_512.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--seq-lens", default=DEFAULT_SEQ_LENS)
    parser.add_argument("--page-size", type=int, default=DSV4_C4_PAGE_SIZE)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()

    (void_atol, void_rtol) = (args.atol, args.rtol)
    _ = (void_atol, void_rtol)

    tokens_list = _parse_tokens(args.tokens)
    seq_lens = _parse_tokens(args.seq_lens)
    _validate_seq_lens(seq_lens)

    ok = True
    case_idx = 0
    _print_header(args)
    for max_seq_len in seq_lens:
        for tokens in tokens_list:
            result = _run_case(tokens, max_seq_len, args, case_idx)
            _print_row(result)
            if result["allclose"] is False or result["sglang_allclose"] is False:
                ok = False
            case_idx += 1
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
