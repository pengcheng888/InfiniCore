import argparse
import statistics
import time

import infinicore
import torch
from infinicore.lib import _infinicore


DSV4_HEAD_DIM = 512
DSV4_NOPE_DIM = 448
DSV4_ROPE_DIM = 64
DSV4_VALUE_BYTES_PER_TOKEN = 576
DSV4_SCALE_BYTES_PER_TOKEN = 8
DSV4_FP8_MAX = 448.0
DSV4_MAX_POSITION_EMBEDDINGS = 1048576
DSV4_RMS_NORM_EPS = 1.0e-6
DSV4_COMPRESS_RATIOS = (
    0,
    0,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    0,
)
DEFAULT_TOKENS = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"
DEFAULT_CASES = "c4,c128"
CASE_TO_PAGE_SIZE = {
    "c4": 64,
    "c128": 2,
}


def _parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _torch_int_dtype(name):
    if name == "int32":
        return torch.int32
    if name == "int64":
        return torch.int64
    raise ValueError(f"unsupported dtype: {name}")


def _page_bytes(page_size):
    raw = (DSV4_VALUE_BYTES_PER_TOKEN + DSV4_SCALE_BYTES_PER_TOKEN) * page_size
    return ((raw + DSV4_VALUE_BYTES_PER_TOKEN - 1) // DSV4_VALUE_BYTES_PER_TOKEN) * DSV4_VALUE_BYTES_PER_TOKEN


def _as_core(tensor, keepalive):
    base = infinicore.from_torch(tensor)
    wrapped = base.as_strided(list(tensor.shape), list(tensor.stride()))
    keepalive.append(base)
    keepalive.append(wrapped)
    return wrapped._underlying


def _sync():
    infinicore.sync_stream()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    _sync()

    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - start) * 1000.0)

    return {
        "avg_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
    }


def _make_freqs(max_pos, device):
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, DSV4_ROPE_DIM, 2, device=device, dtype=torch.float32) / DSV4_ROPE_DIM)
    )
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    angles = torch.outer(t, inv_freq)
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1).flatten(-2).contiguous()


def _make_out_loc(tokens, page_size, out_dtype, invalid_every, device):
    loc = torch.arange(tokens, device=device, dtype=torch.int64)
    if invalid_every > 0:
        loc[invalid_every - 1 :: invalid_every] = -1
    return loc.to(out_dtype).contiguous(), loc.cpu()


def _make_case(tokens, page_size, freqs, pos_dtype, out_dtype, args):
    torch.manual_seed(args.seed + tokens * 17 + page_size)
    device = "cuda"
    if args.strided_batch:
        kv_base = torch.randn((tokens, DSV4_HEAD_DIM + 8), device=device, dtype=torch.bfloat16) * 0.2
        kv = kv_base[:, :DSV4_HEAD_DIM]
    else:
        kv = (torch.randn((tokens, DSV4_HEAD_DIM), device=device, dtype=torch.bfloat16) * 0.2).contiguous()
    baseline_kv = kv.contiguous().clone()
    fused_kv = kv.clone()
    weight = (torch.randn((DSV4_HEAD_DIM,), device=device, dtype=torch.bfloat16) * 0.25).contiguous()
    positions = ((torch.arange(tokens, device=device, dtype=pos_dtype) * 7) % freqs.shape[0]).contiguous()
    out_loc, out_loc_cpu = _make_out_loc(tokens, page_size, out_dtype, args.invalid_every, device)
    blocks = max(1, (tokens + page_size - 1) // page_size)
    baseline_cache = torch.zeros((blocks, _page_bytes(page_size)), device=device, dtype=torch.uint8)
    fused_cache = torch.zeros_like(baseline_cache)

    keepalive = []
    tensors = {
        "kv": kv,
        "baseline_kv": baseline_kv,
        "fused_kv": fused_kv,
        "weight": weight,
        "freqs": freqs,
        "positions": positions,
        "out_loc": out_loc,
        "out_loc_cpu": out_loc_cpu,
        "baseline_cache": baseline_cache,
        "fused_cache": fused_cache,
    }
    core = {
        "baseline_kv": _as_core(baseline_kv, keepalive),
        "fused_kv": _as_core(fused_kv, keepalive),
        "weight": _as_core(weight, keepalive),
        "freqs": _as_core(freqs, keepalive),
        "positions": _as_core(positions, keepalive),
        "out_loc": _as_core(out_loc, keepalive),
        "baseline_cache": _as_core(baseline_cache, keepalive),
        "fused_cache": _as_core(fused_cache, keepalive),
    }
    return tensors, core, keepalive


def _run_baseline(core, eps, page_size):
    _infinicore.deepseek_v4_compress_fused_norm_rope_(
        core["baseline_kv"],
        core["weight"],
        eps,
        core["freqs"],
        core["positions"],
    )
    _infinicore.deepseek_v4_store_flashmla_raw_cache_(
        core["baseline_kv"],
        core["baseline_cache"],
        core["out_loc"],
        page_size,
    )


def _run_fused(core, eps, page_size):
    _infinicore.deepseek_v4_compress_norm_rope_store_(
        core["fused_kv"],
        core["weight"],
        eps,
        core["freqs"],
        core["positions"],
        core["out_loc"],
        core["fused_cache"],
        page_size,
    )


def _compare_cache(tensors, page_size, args):
    flat_baseline = tensors["baseline_cache"].reshape(-1)
    flat_fused = tensors["fused_cache"].reshape(-1)
    non_rope_mask = torch.ones_like(flat_baseline, dtype=torch.bool)
    quant_mask = torch.zeros_like(flat_baseline, dtype=torch.bool)
    quant_rows = []
    rope_rows = []

    for loc in tensors["out_loc_cpu"].tolist():
        if loc < 0:
            continue
        page = loc // page_size
        offset = loc % page_size
        token_base = page * _page_bytes(page_size) + offset * DSV4_VALUE_BYTES_PER_TOKEN
        scale_base = page * _page_bytes(page_size) + DSV4_VALUE_BYTES_PER_TOKEN * page_size + offset * DSV4_SCALE_BYTES_PER_TOKEN
        rope_begin = token_base + DSV4_NOPE_DIM
        rope_end = token_base + DSV4_VALUE_BYTES_PER_TOKEN
        quant_mask[token_base:rope_begin] = True
        quant_mask[scale_base : scale_base + 7] = True
        non_rope_mask[rope_begin:rope_end] = False
        quant_rows.append(
            (
                flat_baseline[token_base:rope_begin],
                flat_baseline[scale_base : scale_base + 7],
                flat_fused[token_base:rope_begin],
                flat_fused[scale_base : scale_base + 7],
            )
        )
        rope_rows.append((flat_baseline[rope_begin:rope_end], flat_fused[rope_begin:rope_end]))

    max_quant_abs = 0.0
    if quant_rows:
        baseline_dequant = []
        fused_dequant = []
        for baseline_values, baseline_scales, fused_values, fused_scales in quant_rows:
            baseline_fp8 = baseline_values.contiguous().view(torch.float8_e4m3fn).float().reshape(7, 64)
            fused_fp8 = fused_values.contiguous().view(torch.float8_e4m3fn).float().reshape(7, 64)
            baseline_scale = torch.pow(
                torch.tensor(2.0, device=baseline_values.device), baseline_scales.float().reshape(7, 1) - 127.0
            )
            fused_scale = torch.pow(
                torch.tensor(2.0, device=fused_values.device), fused_scales.float().reshape(7, 1) - 127.0
            )
            baseline_dequant.append((baseline_fp8 * baseline_scale).reshape(-1))
            fused_dequant.append((fused_fp8 * fused_scale).reshape(-1))
        baseline_dequant = torch.stack(baseline_dequant, dim=0)
        fused_dequant = torch.stack(fused_dequant, dim=0)
        max_quant_abs = (baseline_dequant - fused_dequant).abs().max().item()
        if not torch.allclose(baseline_dequant, fused_dequant, atol=args.quant_atol, rtol=args.quant_rtol):
            raise AssertionError(f"quant dequant mismatch: max_abs={max_quant_abs:.6g}")

    max_rope_abs = 0.0
    if rope_rows:
        baseline_rope = torch.stack([row[0] for row in rope_rows], dim=0).contiguous().view(torch.bfloat16)
        fused_rope = torch.stack([row[1] for row in rope_rows], dim=0).contiguous().view(torch.bfloat16)
        max_rope_abs = (baseline_rope.float() - fused_rope.float()).abs().max().item()
        if not torch.allclose(baseline_rope.float(), fused_rope.float(), atol=args.rope_atol, rtol=args.rope_rtol):
            raise AssertionError(f"RoPE BF16 mismatch: max_abs={max_rope_abs:.6g}")

    strict_mask = non_rope_mask & ~quant_mask
    strict_diff = ((flat_baseline != flat_fused) & strict_mask).sum().item()
    return {
        "strict_diff": strict_diff,
        "max_quant_abs": max_quant_abs,
        "max_rope_abs": max_rope_abs,
    }


def _run_case(case_name, tokens, page_size, freqs, pos_dtype, out_dtype, args):
    tensors, core, keepalive = _make_case(tokens, page_size, freqs, pos_dtype, out_dtype, args)

    check = {
        "strict_diff": -1,
        "max_quant_abs": float("nan"),
        "max_rope_abs": float("nan"),
        "ok": "skip",
    }
    if args.check:
        _run_baseline(core, args.eps, page_size)
        _run_fused(core, args.eps, page_size)
        _sync()
        check.update(_compare_cache(tensors, page_size, args))
        check["ok"] = "True"
        tensors["baseline_kv"].copy_(tensors["kv"])
        tensors["baseline_cache"].zero_()
        tensors["fused_cache"].zero_()
        _sync()

    def baseline_fn():
        _run_baseline(core, args.eps, page_size)

    def fused_fn():
        _run_fused(core, args.eps, page_size)

    baseline = _bench(baseline_fn, args.warmup, args.iters)
    fused = _bench(fused_fn, args.warmup, args.iters)
    del keepalive
    return {
        "case": case_name,
        "tokens": tokens,
        "page_size": page_size,
        "cache_pages": tensors["baseline_cache"].shape[0],
        "cache_page_bytes": tensors["baseline_cache"].shape[1],
        "baseline": baseline,
        "fused": fused,
        "speedup_avg": baseline["avg_ms"] / fused["avg_ms"] if fused["avg_ms"] > 0 else float("inf"),
        "speedup_median": baseline["median_ms"] / fused["median_ms"] if fused["median_ms"] > 0 else float("inf"),
        "check": check,
    }


def _format_float(value):
    return "nan" if value != value else f"{value:.4e}"


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DeepSeek-V4 compress_norm_rope_store against the current two-op InfiniLM path."
    )
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--cases", default=DEFAULT_CASES, help="Comma-separated list from: c4,c128")
    parser.add_argument("--max-pos", type=int, default=DSV4_MAX_POSITION_EMBEDDINGS)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--eps", type=float, default=DSV4_RMS_NORM_EPS)
    parser.add_argument("--pos-dtype", choices=("int32", "int64"), default="int64")
    parser.add_argument("--out-dtype", choices=("int32", "int64"), default="int32")
    parser.add_argument("--invalid-every", type=int, default=0)
    parser.add_argument("--strided-batch", action="store_true")
    parser.add_argument("--check", action="store_true", default=True)
    parser.add_argument("--no-check", dest="check", action="store_false")
    parser.add_argument("--rope-atol", type=float, default=2.0e-2)
    parser.add_argument("--rope-rtol", type=float, default=2.0e-2)
    parser.add_argument("--quant-atol", type=float, default=3.0e-1)
    parser.add_argument("--quant-rtol", type=float, default=2.0e-2)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()

    cases = [item.strip().lower() for item in args.cases.split(",") if item.strip()]
    tokens_list = _parse_int_list(args.tokens)
    pos_dtype = _torch_int_dtype(args.pos_dtype)
    out_dtype = _torch_int_dtype(args.out_dtype)
    freqs = _make_freqs(args.max_pos, "cuda")
    c4_layers = sum(1 for ratio in DSV4_COMPRESS_RATIOS if ratio == 4)
    c128_layers = sum(1 for ratio in DSV4_COMPRESS_RATIOS if ratio == 128)

    print("DeepSeek-V4 compress_norm_rope_store 性能测试")
    print(
        f"head_dim={DSV4_HEAD_DIM} nope_dim={DSV4_NOPE_DIM} rope_dim={DSV4_ROPE_DIM} "
        f"max_position_embeddings={args.max_pos} eps={args.eps}"
    )
    print(
        f"compress_ratios: c4_layers={c4_layers} c128_layers={c128_layers} "
        f"page_size(c4)=64 page_size(c128)=2"
    )
    print(
        f"tokens={args.tokens} cases={args.cases} pos_dtype={args.pos_dtype} out_dtype={args.out_dtype} "
        f"iters={args.iters} warmup={args.warmup} check={args.check}"
    )
    print(
        f"{'case':>5} | {'tokens':>6} | {'page':>4} | {'pages':>6} | {'page_bytes':>10} | "
        f"{'baseline avg':>12} | {'fused avg':>10} | {'avg spd':>7} | "
        f"{'baseline med':>12} | {'fused med':>10} | {'med spd':>7} | "
        f"{'q_abs':>10} | {'rope_abs':>10} | {'strict':>6} | {'ok':>5}"
    )
    print("-" * 151)
    for case_name in cases:
        if case_name not in CASE_TO_PAGE_SIZE:
            raise ValueError(f"unsupported case: {case_name}")
        page_size = CASE_TO_PAGE_SIZE[case_name]
        for tokens in tokens_list:
            result = _run_case(case_name, tokens, page_size, freqs, pos_dtype, out_dtype, args)
            baseline = result["baseline"]
            fused = result["fused"]
            check = result["check"]
            print(
                f"{result['case']:>5} | {result['tokens']:6d} | {result['page_size']:4d} | "
                f"{result['cache_pages']:6d} | {result['cache_page_bytes']:10d} | "
                f"{baseline['avg_ms']:12.4f} | {fused['avg_ms']:10.4f} | {result['speedup_avg']:7.2f} | "
                f"{baseline['median_ms']:12.4f} | {fused['median_ms']:10.4f} | {result['speedup_median']:7.2f} | "
                f"{_format_float(check['max_quant_abs']):>10} | {_format_float(check['max_rope_abs']):>10} | "
                f"{check['strict_diff']:6d} | {check['ok']:>5}"
            )


if __name__ == "__main__":
    main()
