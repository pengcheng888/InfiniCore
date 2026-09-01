import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
from infinicore.lib import _infinicore


def _as_core(tensor):
    return infinicore.from_torch(tensor).as_strided(list(tensor.shape), list(tensor.stride()))


def _make_inputs(args, device):
    torch.manual_seed(args.seed)
    n_blocks = max((args.cache_seqlen + args.page_size - 1) // args.page_size, 1)
    q = (torch.randn((args.batch, args.seq_q, args.q_heads, args.head_dim), device=device, dtype=torch.bfloat16) / 10).contiguous()
    k_cache = (torch.randn((n_blocks, args.page_size, args.kv_heads, args.head_dim), device=device, dtype=torch.bfloat16) / 10).contiguous()
    cache_seqlens = torch.full((args.batch,), args.cache_seqlen, device=device, dtype=torch.int32).contiguous()
    block_table = torch.arange(n_blocks, device=device, dtype=torch.int32).reshape(1, -1).repeat(args.batch, 1).contiguous()
    ref_out = torch.empty((args.batch, args.seq_q, args.q_heads, args.head_size_v), device=device, dtype=torch.bfloat16)
    ref_lse = torch.empty((args.batch, args.q_heads, args.seq_q), device=device, dtype=torch.float32)
    return {
        "q": q,
        "k_cache": k_cache,
        "cache_seqlens": cache_seqlens,
        "block_table": block_table,
        "ref_out": ref_out,
        "ref_lse": ref_lse,
    }


def _run_flash_mla_ref(tensors, args, flash_mla_cuda):
    if hasattr(flash_mla_cuda, "dense_decode_fwd"):
        out, lse, _, _ = flash_mla_cuda.dense_decode_fwd(
            tensors["q"],
            tensors["k_cache"],
            args.head_size_v,
            tensors["cache_seqlens"],
            tensors["block_table"],
            args.softmax_scale,
            args.causal,
            None,
            None,
        )
    else:
        num_heads_per_head_k = args.seq_q * args.q_heads // args.kv_heads
        tile_scheduler_metadata, num_splits = flash_mla_cuda.get_mla_metadata(
            tensors["cache_seqlens"],
            num_heads_per_head_k,
            args.kv_heads,
        )
        out, lse = flash_mla_cuda.fwd_kvcache_mla(
            tensors["q"],
            tensors["k_cache"],
            None,
            args.head_size_v,
            tensors["cache_seqlens"],
            tensors["block_table"],
            args.softmax_scale,
            args.causal,
            tile_scheduler_metadata,
            num_splits,
        )
    tensors["ref_out"].copy_(out)
    tensors["ref_lse"].copy_(lse)
    return tensors["ref_out"], tensors["ref_lse"]


def _run_infinicore(core, args):
    out, lse, _, _ = _infinicore.flash_mla_dense_decode_fwd(
        core["q"]._underlying,
        core["k_cache"]._underlying,
        args.head_size_v,
        core["cache_seqlens"]._underlying,
        core["block_table"]._underlying,
        args.softmax_scale,
        args.causal,
        None,
        None,
    )
    return out, lse


def _max_diff(lhs, rhs):
    lhs_f = lhs.float()
    rhs_f = rhs.float()
    abs_diff = (lhs_f - rhs_f).abs()
    max_abs = abs_diff.max().item() if abs_diff.numel() > 0 else 0.0
    max_rel = (abs_diff / rhs_f.abs().clamp_min(1e-6)).max().item() if abs_diff.numel() > 0 else 0.0
    return max_abs, max_rel


def _bench(fn, warmup, iters):
    value = None
    for _ in range(warmup):
        value = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        value = fn()
    end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    return value, total_ms / iters, total_ms


def main():
    parser = argparse.ArgumentParser(description="Test DeepSeek-V4 FlashMLA dense_decode_fwd bridge.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--metax", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-q", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=576)
    parser.add_argument("--head-size-v", type=int, default=512)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--cache-seqlen", type=int, default=512)
    parser.add_argument("--softmax-scale", type=float, default=1.0 / (576 ** 0.5))
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()

    try:
        import flash_mla.cuda as flash_mla_cuda
    except ImportError:
        import flash_mla_cuda

    device = torch.device("cuda")
    tensors = _make_inputs(args, device)
    core = {name: _as_core(tensor) for name, tensor in tensors.items() if isinstance(tensor, torch.Tensor)}

    torch.cuda.synchronize()

    (ref_out, ref_lse), ref_avg, ref_total = _bench(lambda: _run_flash_mla_ref(tensors, args, flash_mla_cuda), args.warmup, args.iters)
    (out, lse), op_avg, op_total = _bench(lambda: _run_infinicore(core, args), args.warmup, args.iters)

    out_abs, out_rel = _max_diff(out, ref_out)
    lse_abs, lse_rel = _max_diff(lse, ref_lse)
    out_ok = torch.equal(out, ref_out) if args.atol == 0 and args.rtol == 0 else torch.allclose(out.float(), ref_out.float(), atol=args.atol, rtol=args.rtol)
    lse_ok = torch.equal(lse, ref_lse) if args.atol == 0 and args.rtol == 0 else torch.allclose(lse.float(), ref_lse.float(), atol=args.atol, rtol=args.rtol)

    print(f"shape q={tuple(tensors['q'].shape)} k_cache={tuple(tensors['k_cache'].shape)} block_table={tuple(tensors['block_table'].shape)} out={tuple(out.shape)} lse={tuple(lse.shape)}")
    print(f"reference avg_ms={ref_avg:.4f} total_ms={ref_total:.4f}")
    print(f"infinicore avg_ms={op_avg:.4f} total_ms={op_total:.4f}")
    print(f"out max_abs={out_abs:.6g} max_rel={out_rel:.6g} equal={out_ok}")
    print(f"lse max_abs={lse_abs:.6g} max_rel={lse_rel:.6g} equal={lse_ok}")

    if not (out_ok and lse_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
