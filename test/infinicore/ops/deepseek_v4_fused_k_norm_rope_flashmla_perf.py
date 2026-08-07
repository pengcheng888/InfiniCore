import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
from infinicore.lib import _infinicore
import torch


DEFAULT_TOKENS = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"
DEFAULT_HEAD_DIM = 512
DEFAULT_ROPE_DIM = 64
DEFAULT_PAGE_SIZE = 256
DEFAULT_MAX_POS = 1048576
DEFAULT_EPS = 1e-6
DEFAULT_POS_DTYPE = "int64"
DEFAULT_OUT_DTYPE = "int32"


def _parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _torch_int_dtype(name):
    if name == "int32":
        return torch.int32
    if name == "int64":
        return torch.int64
    raise ValueError(f"unsupported dtype: {name}")


def _page_bytes(page_size):
    return ((584 * page_size + 575) // 576) * 576


def _as_core(tensor, keepalive):
    base = infinicore.from_torch(tensor)
    wrapped = base.as_strided(list(tensor.shape), list(tensor.stride()))
    keepalive.append(base)
    keepalive.append(wrapped)
    return wrapped._underlying


def _sync():
    infinicore.sync_stream()


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
    total_ms = sum(samples)
    return {
        "total_ms": total_ms,
        "avg_ms": total_ms / float(iters),
        "median_ms": statistics.median(samples),
    }


def _make_freqs(max_pos, dim=DEFAULT_ROPE_DIM, device="cuda"):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    angles = torch.outer(t, inv_freq)
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1).flatten(-2).contiguous()


def _make_out_loc(tokens, page_size, out_dtype, invalid_every):
    loc = torch.arange(tokens, device="cuda", dtype=torch.int64)
    if invalid_every > 0:
        loc[invalid_every - 1 :: invalid_every] = -1
    return loc.to(out_dtype), loc.cpu()


def _reference_cache(kv, weight, eps, freqs_cis, positions, out_loc_cpu, blocks, page_size):
    tokens = kv.shape[0]
    page_bytes = _page_bytes(page_size)
    ref = torch.zeros((blocks, page_bytes), device=kv.device, dtype=torch.uint8)

    x = kv.float()
    norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight.float()

    no_pe = norm[:, :448].reshape(tokens, 7, 64)
    scale_raw = torch.clamp(no_pe.abs().amax(dim=-1, keepdim=True), min=1.0e-4) / 448.0
    scale_exp = torch.clamp(torch.ceil(torch.log2(scale_raw)).to(torch.int32) + 127, 0, 255).to(torch.uint8)
    scale = torch.pow(torch.tensor(2.0, device=kv.device), scale_exp.float() - 127.0)
    quant = torch.clamp(no_pe / scale, -448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8).reshape(tokens, 448)

    tail = norm[:, 448:].reshape(tokens, 32, 2)
    freqs = freqs_cis.index_select(0, positions.long()).reshape(tokens, 32, 2)
    c = freqs[..., 0]
    s = freqs[..., 1]
    xr = tail[..., 0]
    xi = tail[..., 1]
    rope = torch.stack((xr * c - xi * s, xr * s + xi * c), dim=-1).reshape(tokens, 64)
    rope_bytes = rope.to(torch.bfloat16).contiguous().view(torch.uint8).reshape(tokens, 128)

    flat = ref.reshape(-1)
    for row, loc in enumerate(out_loc_cpu.tolist()):
        if loc < 0:
            continue
        page = loc // page_size
        offset = loc % page_size
        token_base = page * page_bytes + offset * 576
        scale_base = page * page_bytes + 576 * page_size + offset * 8
        flat[token_base : token_base + 448] = quant[row]
        flat[token_base + 448 : token_base + 576] = rope_bytes[row]
        flat[scale_base : scale_base + 7] = scale_exp[row].reshape(7)
    return ref


def _make_case(tokens, freqs, pos_dtype, out_dtype, args):
    torch.manual_seed(args.seed + tokens * 17)
    if args.strided_batch:
        base = torch.randn((tokens, DEFAULT_HEAD_DIM + 8), device="cuda", dtype=torch.bfloat16)
        kv = base[:, :DEFAULT_HEAD_DIM]
    else:
        kv = torch.randn((tokens, DEFAULT_HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    baseline_kv = kv.clone()
    fused_kv = kv.clone()
    weight = torch.randn((DEFAULT_HEAD_DIM,), device="cuda", dtype=torch.bfloat16)
    positions = ((torch.arange(tokens, device="cuda", dtype=pos_dtype) * 5) % freqs.shape[0]).contiguous()
    out_loc, out_loc_cpu = _make_out_loc(tokens, args.page_size, out_dtype, args.invalid_every)
    valid_tokens = tokens - (tokens // args.invalid_every if args.invalid_every > 0 else 0)
    blocks = max(1, (tokens + args.page_size - 1) // args.page_size)
    baseline_cache = torch.zeros((blocks, _page_bytes(args.page_size)), device="cuda", dtype=torch.uint8)
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
        "valid_tokens": valid_tokens,
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
    _infinicore.deepseek_v4_fused_norm_rope_inplace_kernel_(
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
    _infinicore.deepseek_v4_fused_k_norm_rope_flashmla_(
        core["fused_kv"],
        core["weight"],
        eps,
        core["freqs"],
        core["positions"],
        core["out_loc"],
        core["fused_cache"],
        page_size,
    )


def _check_fused(tensors, args):
    ref = _reference_cache(
        tensors["kv"],
        tensors["weight"],
        args.eps,
        tensors["freqs"],
        tensors["positions"],
        tensors["out_loc_cpu"],
        tensors["fused_cache"].shape[0],
        args.page_size,
    )
    tensors["fused_cache"].zero_()
    _sync()
    keepalive = []
    core = {
        "fused_kv": _as_core(tensors["fused_kv"], keepalive),
        "weight": _as_core(tensors["weight"], keepalive),
        "freqs": _as_core(tensors["freqs"], keepalive),
        "positions": _as_core(tensors["positions"], keepalive),
        "out_loc": _as_core(tensors["out_loc"], keepalive),
        "fused_cache": _as_core(tensors["fused_cache"], keepalive),
    }
    _run_fused(core, args.eps, args.page_size)
    _sync()
    fused_cache = tensors["fused_cache"]
    page_bytes = _page_bytes(args.page_size)
    flat_fused = fused_cache.reshape(-1)
    flat_ref = ref.reshape(-1)
    non_rope_mask = torch.ones_like(flat_ref, dtype=torch.bool)
    rope_fused_rows = []
    rope_ref_rows = []
    for loc in tensors["out_loc_cpu"].tolist():
        if loc < 0:
            continue
        page = loc // args.page_size
        offset = loc % args.page_size
        token_base = page * page_bytes + offset * 576
        rope_begin = token_base + 448
        rope_end = token_base + 576
        non_rope_mask[rope_begin:rope_end] = False
        rope_fused_rows.append(flat_fused[rope_begin:rope_end])
        rope_ref_rows.append(flat_ref[rope_begin:rope_end])

    non_rope_diff = ((flat_fused != flat_ref) & non_rope_mask).sum().item()
    if non_rope_diff != 0:
        raise AssertionError(f"fused cache mismatch outside RoPE BF16 region: diff_bytes={non_rope_diff}")

    rope_max_abs = 0.0
    if rope_fused_rows:
        rope_fused = torch.stack(rope_fused_rows, dim=0).contiguous().view(torch.bfloat16).reshape(-1, DEFAULT_ROPE_DIM)
        rope_ref = torch.stack(rope_ref_rows, dim=0).contiguous().view(torch.bfloat16).reshape(-1, DEFAULT_ROPE_DIM)
        rope_diff = (rope_fused.float() - rope_ref.float()).abs()
        rope_max_abs = rope_diff.max().item()
        if not torch.allclose(rope_fused.float(), rope_ref.float(), rtol=args.rope_rtol, atol=args.rope_atol):
            raise AssertionError(
                "fused cache RoPE BF16 mismatch: "
                f"max_abs={rope_max_abs:.6g}, atol={args.rope_atol}, rtol={args.rope_rtol}"
            )
    del keepalive
    return {"non_rope_diff": non_rope_diff, "rope_max_abs": rope_max_abs}


def _run_case(tokens, freqs, pos_dtype, out_dtype, args):
    tensors, core, keepalive = _make_case(tokens, freqs, pos_dtype, out_dtype, args)

    checked = "skip"
    if args.check and tokens <= args.check_max_tokens:
        check_result = _check_fused(tensors, args)
        checked = f"ok/{check_result['rope_max_abs']:.3g}"
        tensors["baseline_kv"].copy_(tensors["kv"])
        tensors["fused_kv"].copy_(tensors["kv"])
        tensors["baseline_cache"].zero_()
        tensors["fused_cache"].zero_()
        _sync()

    def baseline_fn():
        _run_baseline(core, args.eps, args.page_size)

    def fused_fn():
        _run_fused(core, args.eps, args.page_size)

    baseline = _bench(baseline_fn, args.warmup, args.iters)
    fused = _bench(fused_fn, args.warmup, args.iters)
    del keepalive
    return {
        "tokens": tokens,
        "valid_tokens": tensors["valid_tokens"],
        "stride0": tensors["fused_kv"].stride(0),
        "baseline": baseline,
        "fused": fused,
        "speedup_avg": baseline["avg_ms"] / fused["avg_ms"] if fused["avg_ms"] > 0 else float("inf"),
        "speedup_median": baseline["median_ms"] / fused["median_ms"] if fused["median_ms"] > 0 else float("inf"),
        "checked": checked,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark DeepSeek-V4 fused_k_norm_rope_flashmla over DEFAULT_TOKENS.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--max-pos", type=int, default=DEFAULT_MAX_POS)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--pos-dtype", choices=("int32", "int64"), default=DEFAULT_POS_DTYPE)
    parser.add_argument("--out-dtype", choices=("int32", "int64"), default=DEFAULT_OUT_DTYPE)
    parser.add_argument("--strided-batch", action="store_true")
    parser.add_argument("--invalid-every", type=int, default=0)
    parser.add_argument("--check", action="store_true", default=True)
    parser.add_argument("--no-check", dest="check", action="store_false")
    parser.add_argument("--check-max-tokens", type=int, default=512)
    parser.add_argument("--rope-atol", type=float, default=2.0e-2)
    parser.add_argument("--rope-rtol", type=float, default=2.0e-2)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()

    tokens_list = _parse_int_list(args.tokens)
    pos_dtype = _torch_int_dtype(args.pos_dtype)
    out_dtype = _torch_int_dtype(args.out_dtype)
    freqs = _make_freqs(args.max_pos, device="cuda")

    print("DeepSeek-V4 fused_k_norm_rope_flashmla 性能测试")
    print(
        f"tokens={args.tokens} head_dim={DEFAULT_HEAD_DIM} rope_dim={DEFAULT_ROPE_DIM} "
        f"page_size={args.page_size} max_pos={args.max_pos} eps={args.eps} "
        f"pos_dtype={args.pos_dtype} out_dtype={args.out_dtype} "
        f"iters={args.iters} warmup={args.warmup} strided_batch={args.strided_batch} "
        f"invalid_every={args.invalid_every} check={args.check} check_max_tokens={args.check_max_tokens} "
        f"rope_atol={args.rope_atol} rope_rtol={args.rope_rtol}"
    )
    print(
        f"{'tokens':>8} | {'valid':>8} | {'stride0':>8} | "
        f"{'base avg':>10} | {'base med':>10} | "
        f"{'fused avg':>10} | {'fused med':>10} | "
        f"{'spd avg':>8} | {'spd med':>8} | {'checked':>8}"
    )
    print("-" * 116)
    for tokens in tokens_list:
        result = _run_case(tokens, freqs, pos_dtype, out_dtype, args)
        baseline = result["baseline"]
        fused = result["fused"]
        print(
            f"{result['tokens']:8d} | "
            f"{result['valid_tokens']:8d} | "
            f"{result['stride0']:8d} | "
            f"{baseline['avg_ms']:10.4f} | "
            f"{baseline['median_ms']:10.4f} | "
            f"{fused['avg_ms']:10.4f} | "
            f"{fused['median_ms']:10.4f} | "
            f"{result['speedup_avg']:8.2f} | "
            f"{result['speedup_median']:8.2f} | "
            f"{str(result['checked']):>8}"
        )


if __name__ == "__main__":
    main()
