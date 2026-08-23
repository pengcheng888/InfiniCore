import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
from infinicore.lib import _infinicore


DEFAULT_TOKENS = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"
DEFAULT_HEAD_DIM = 512
DEFAULT_ROPE_DIM = 64
DEFAULT_PAGE_SIZE = 256
DEFAULT_MAX_POS = 1048576


def _parse_tokens(text):
    return [int(item) for item in text.split(",") if item.strip()]


def _torch_int_dtype(name):
    if name == "int32":
        return torch.int32
    if name == "int64":
        return torch.int64
    raise ValueError(f"unsupported dtype: {name}")


def _page_bytes(page_size):
    return ((584 * page_size + 575) // 576) * 576


def _as_core(tensor):
    return infinicore.from_torch(tensor).as_strided(list(tensor.shape), list(tensor.stride()))


def _make_freqs(max_pos, dim=DEFAULT_ROPE_DIM, device="cuda"):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    angles = torch.outer(t, inv_freq)
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1).flatten(-2).contiguous()


def _make_out_loc(tokens, page_size, out_dtype, invalid_every, device):
    loc = torch.arange(tokens, device=device, dtype=torch.int64)
    if invalid_every > 0:
        loc[invalid_every - 1 :: invalid_every] = -1
    return loc.to(out_dtype), loc.cpu()


def _make_inputs(tokens, freqs, pos_dtype, out_dtype, args, device):
    torch.manual_seed(args.seed + tokens * 17)
    if args.strided_batch:
        base = torch.randn((tokens, DEFAULT_HEAD_DIM * 3), device=device, dtype=torch.bfloat16)
        kv = base[:, 128 : 128 + DEFAULT_HEAD_DIM]
    else:
        kv = torch.randn((tokens, DEFAULT_HEAD_DIM), device=device, dtype=torch.bfloat16).contiguous()
    ref_kv = torch.empty_like(kv)
    out_kv = torch.empty_like(kv)
    weight = torch.randn((DEFAULT_HEAD_DIM,), device=device, dtype=torch.bfloat16).contiguous()
    positions = ((torch.arange(tokens, device=device, dtype=pos_dtype) * 5) % freqs.shape[0]).contiguous()
    out_loc, out_loc_cpu = _make_out_loc(tokens, args.page_size, out_dtype, args.invalid_every, device)
    blocks = max(1, (tokens + args.page_size - 1) // args.page_size)
    ref_cache = torch.zeros((blocks, _page_bytes(args.page_size)), device=device, dtype=torch.uint8)
    out_cache = torch.zeros_like(ref_cache)
    valid_tokens = tokens - (tokens // args.invalid_every if args.invalid_every > 0 else 0)
    return {
        "kv": kv,
        "ref_kv": ref_kv,
        "out_kv": out_kv,
        "weight": weight,
        "freqs": freqs,
        "positions": positions,
        "out_loc": out_loc,
        "out_loc_cpu": out_loc_cpu,
        "ref_cache": ref_cache,
        "out_cache": out_cache,
        "valid_tokens": valid_tokens,
    }


def _ref_separate(core, tensors, eps, page_size):
    tensors["ref_kv"].copy_(tensors["kv"])
    tensors["ref_cache"].zero_()
    _infinicore.deepseek_v4_fused_norm_rope_inplace_kernel_(
        core["ref_kv"]._underlying,
        core["weight"]._underlying,
        eps,
        core["freqs"]._underlying,
        core["positions"]._underlying,
    )
    _infinicore.deepseek_v4_store_flashmla_raw_cache_(
        core["ref_kv"]._underlying,
        core["ref_cache"]._underlying,
        core["out_loc"]._underlying,
        page_size,
    )
    return tensors["ref_cache"]


def _run_op(core, tensors, eps, page_size):
    tensors["out_kv"].copy_(tensors["kv"])
    tensors["out_cache"].zero_()
    _infinicore.deepseek_v4_fused_k_norm_rope_flashmla_(
        core["out_kv"]._underlying,
        core["weight"]._underlying,
        eps,
        core["freqs"]._underlying,
        core["positions"]._underlying,
        core["out_loc"]._underlying,
        core["out_cache"]._underlying,
        page_size,
    )
    return tensors["out_cache"]


def _cache_diff(got, ref, out_loc_cpu, page_size, rope_atol, rope_rtol, quant_atol, quant_rtol):
    page_bytes = _page_bytes(page_size)
    flat_got = got.reshape(-1)
    flat_ref = ref.reshape(-1)
    non_rope_mask = torch.ones_like(flat_ref, dtype=torch.bool)
    quant_mask = torch.zeros_like(flat_ref, dtype=torch.bool)
    quant_got_rows = []
    quant_ref_rows = []
    rope_got_rows = []
    rope_ref_rows = []
    for loc in out_loc_cpu.tolist():
        if loc < 0:
            continue
        page = loc // page_size
        offset = loc % page_size
        token_base = page * page_bytes + offset * 576
        scale_base = page * page_bytes + 576 * page_size + offset * 8
        rope_begin = token_base + 448
        rope_end = token_base + 576
        quant_mask[token_base:rope_begin] = True
        quant_mask[scale_base : scale_base + 7] = True
        non_rope_mask[rope_begin:rope_end] = False
        quant_got_rows.append((flat_got[token_base:rope_begin], flat_got[scale_base : scale_base + 7]))
        quant_ref_rows.append((flat_ref[token_base:rope_begin], flat_ref[scale_base : scale_base + 7]))
        rope_got_rows.append(flat_got[rope_begin:rope_end])
        rope_ref_rows.append(flat_ref[rope_begin:rope_end])

    strict_mask = non_rope_mask & ~quant_mask
    strict_diff = ((flat_got != flat_ref) & strict_mask).sum().item()
    max_abs = 0.0
    max_rel = 0.0
    quant_ok = True
    if quant_got_rows:
        quant_got = []
        quant_ref = []
        for (got_values, got_scales), (ref_values, ref_scales) in zip(quant_got_rows, quant_ref_rows):
            got_fp8 = got_values.contiguous().view(torch.float8_e4m3fn).float().reshape(7, 64)
            ref_fp8 = ref_values.contiguous().view(torch.float8_e4m3fn).float().reshape(7, 64)
            got_scale = torch.pow(torch.tensor(2.0, device=got.device), got_scales.float().reshape(7, 1) - 127.0)
            ref_scale = torch.pow(torch.tensor(2.0, device=ref.device), ref_scales.float().reshape(7, 1) - 127.0)
            quant_got.append((got_fp8 * got_scale).reshape(-1))
            quant_ref.append((ref_fp8 * ref_scale).reshape(-1))
        quant_got = torch.stack(quant_got, dim=0)
        quant_ref = torch.stack(quant_ref, dim=0)
        quant_abs = (quant_got - quant_ref).abs()
        max_abs = max(max_abs, quant_abs.max().item() if quant_abs.numel() > 0 else 0.0)
        max_rel = max(
            max_rel,
            (quant_abs / quant_ref.abs().clamp_min(1e-6)).max().item() if quant_abs.numel() > 0 else 0.0,
        )
        quant_ok = torch.allclose(quant_got, quant_ref, atol=quant_atol, rtol=quant_rtol)

    rope_ok = True
    if rope_got_rows:
        rope_got = torch.stack(rope_got_rows, dim=0).contiguous().view(torch.bfloat16).reshape(-1, DEFAULT_ROPE_DIM)
        rope_ref = torch.stack(rope_ref_rows, dim=0).contiguous().view(torch.bfloat16).reshape(-1, DEFAULT_ROPE_DIM)
        abs_diff = (rope_got.float() - rope_ref.float()).abs()
        max_abs = max(max_abs, abs_diff.max().item() if abs_diff.numel() > 0 else 0.0)
        max_rel = max(
            max_rel,
            (abs_diff / rope_ref.float().abs().clamp_min(1e-6)).max().item() if abs_diff.numel() > 0 else 0.0,
        )
        rope_ok = torch.allclose(rope_got.float(), rope_ref.float(), atol=rope_atol, rtol=rope_rtol)
    return max_abs, max_rel, strict_diff == 0 and quant_ok and rope_ok


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


def _run_case(tokens, freqs, pos_dtype, out_dtype, args):
    device = torch.device("cuda")
    tensors = _make_inputs(tokens, freqs, pos_dtype, out_dtype, args, device)
    core = {name: _as_core(tensor) for name, tensor in tensors.items() if isinstance(tensor, torch.Tensor) and tensor.is_cuda}

    ref_perf = _bench(lambda: _ref_separate(core, tensors, args.eps, args.page_size), args.warmup, args.iters)
    op_perf = _bench(lambda: _run_op(core, tensors, args.eps, args.page_size), args.warmup, args.iters)

    max_abs, max_rel, allclose = _cache_diff(
        op_perf["warmup_value"],
        ref_perf["warmup_value"],
        tensors["out_loc_cpu"],
        args.page_size,
        args.rope_atol,
        args.rope_rtol,
        args.quant_atol,
        args.quant_rtol,
    )
    if not allclose:
        print(
            f"[FAIL] tokens={tokens} stride0={tensors['kv'].stride(0)} "
            f"max_abs={max_abs:.6g} max_rel={max_rel:.6g}"
        )

    return {
        "tokens": tokens,
        "valid_tokens": tensors["valid_tokens"],
        "stride0": tensors["kv"].stride(0),
        "ref_avg": ref_perf["avg_ms"],
        "op_avg": op_perf["avg_ms"],
        "speedup": ref_perf["avg_ms"] / op_perf["avg_ms"] if op_perf["avg_ms"] > 0 else float("inf"),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "allclose": allclose,
    }


def _print_header(args):
    print(
        f"\nhead_dim={DEFAULT_HEAD_DIM} rope_dim={DEFAULT_ROPE_DIM} page_size={args.page_size} "
        f"pos_dtype={args.pos_dtype} out_dtype={args.out_dtype} strided_batch={args.strided_batch}"
    )
    print(
        f"{'tokens':>8} | {'valid':>8} | {'stride0':>8} | {'ref avg':>10} | "
        f"{'op avg':>10} | {'speedup':>8} | {'max_abs':>10} | {'max_rel':>10} | {'allclose':>8}"
    )
    print("-" * 110)


def _print_row(result):
    print(
        f"{result['tokens']:8d} | "
        f"{result['valid_tokens']:8d} | "
        f"{result['stride0']:8d} | "
        f"{result['ref_avg']:10.4f} | "
        f"{result['op_avg']:10.4f} | "
        f"{result['speedup']:8.2f} | "
        f"{result['max_abs']:10.6g} | "
        f"{result['max_rel']:10.6g} | "
        f"{str(result['allclose']):>8}"
    )


def main():
    parser = argparse.ArgumentParser(description="Check and benchmark DeepSeek-V4 fused_k_norm_rope_flashmla.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--max-pos", type=int, default=DEFAULT_MAX_POS)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--pos-dtype", choices=["int32", "int64"], default="int64")
    parser.add_argument("--out-dtype", choices=["int32", "int64"], default="int32")
    parser.add_argument("--strided-batch", action="store_true")
    parser.add_argument("--invalid-every", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--rope-atol", type=float, default=2e-2)
    parser.add_argument("--rope-rtol", type=float, default=2e-2)
    parser.add_argument("--quant-atol", type=float, default=1.1)
    parser.add_argument("--quant-rtol", type=float, default=2e-2)
    args = parser.parse_args()

    pos_dtype = _torch_int_dtype(args.pos_dtype)
    out_dtype = _torch_int_dtype(args.out_dtype)
    freqs = _make_freqs(args.max_pos, device="cuda")

    ok = True
    _print_header(args)
    for tokens in _parse_tokens(args.tokens):
        result = _run_case(tokens, freqs, pos_dtype, out_dtype, args)
        _print_row(result)
        if result["allclose"] is False:
            ok = False

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
