import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
from infinicore.lib import _infinicore


def _as_core(tensor):
    return infinicore.from_torch(tensor).as_strided(list(tensor.shape), list(tensor.stride()))


def _quantize_k_cache_v32(input_k_cache):
    num_blocks, block_size, h_k, d = input_k_cache.shape
    assert h_k == 1
    assert d == 576
    d_nope, d_rope, tile_size, num_tiles = 512, 64, 128, 4
    bytes_per_token = d_nope + num_tiles * 4 + input_k_cache.element_size() * d_rope
    src = input_k_cache[:, :, 0, :]
    result = torch.empty(
        (num_blocks, block_size + 1, bytes_per_token),
        dtype=torch.float8_e4m3fn,
        device=input_k_cache.device,
    )[:, :block_size, :]
    nope = result[..., :d_nope]
    scales = result[..., d_nope : d_nope + num_tiles * 4].view(torch.float32)
    rope = result[..., d_nope + num_tiles * 4 :].view(torch.bfloat16)
    rope.copy_(src[..., d_nope:])
    for tile in range(num_tiles):
        start = tile * tile_size
        end = start + tile_size
        scale = torch.abs(src[..., start:end]).max(dim=-1).values.float() / 448.0
        scale = torch.pow(2, torch.clamp_min(scale, 1e-4).log2().ceil())
        scales[:, :, tile] = scale
        nope[..., start:end] = (src[..., start:end].float() / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return result.view(num_blocks, block_size, 1, bytes_per_token)


def _make_inputs(args, device):
    if args.mode == "sparse" and args.seq_q != 1:
        raise ValueError("sparse FlashMLA decode test expects --seq-q 1.")
    torch.manual_seed(args.seed)
    n_blocks = max((args.cache_seqlen + args.page_size - 1) // args.page_size, 1)
    total_slots = n_blocks * args.page_size
    q = (torch.randn((args.batch, args.seq_q, args.q_heads, args.head_dim), device=device, dtype=torch.bfloat16) / 10).contiguous()
    k_cache_bf16 = (torch.randn((n_blocks, args.page_size, args.kv_heads, args.head_dim), device=device, dtype=torch.bfloat16) / 10).contiguous()
    if args.mode == "sparse" and not args.bf16_kvcache:
        k_cache = _quantize_k_cache_v32(k_cache_bf16).contiguous()
    else:
        k_cache = k_cache_bf16
    indices = (torch.arange(args.topk, device=device, dtype=torch.int32) % total_slots).reshape(1, 1, -1)
    indices = indices.repeat(args.batch, args.seq_q, 1).contiguous()
    topk_length = torch.full((args.batch,), args.topk, device=device, dtype=torch.int32).contiguous()
    ref_out = torch.empty((args.batch, args.seq_q, args.q_heads, args.head_size_v), device=device, dtype=torch.bfloat16)
    ref_lse = torch.empty((args.batch, args.q_heads, args.seq_q), device=device, dtype=torch.float32)
    return {
        "q": q,
        "k_cache": k_cache,
        "indices": indices,
        "topk_length": topk_length,
        "ref_out": ref_out,
        "ref_lse": ref_lse,
    }


def _run_flash_mla_ref(tensors, args, flash_mla_cuda):
    result = flash_mla_cuda.sparse_decode_fwd(
        tensors["q"],
        tensors["k_cache"],
        tensors["indices"],
        None,  # topk_length
        None,  # attn_sink
        None,  # tile_scheduler_metadata
        None,  # num_splits
        None,  # extra_k_cache
        None,  # extra_indices_in_kvcache
        None,  # extra_topk_length
        args.head_size_v,
        args.softmax_scale,
    )
    tensors["ref_out"].copy_(result[0])
    tensors["ref_lse"].copy_(result[1])
    return tensors["ref_out"], tensors["ref_lse"]


def _infer_num_sm_parts(tensors, args, flash_mla_cuda):
    if args.num_sm_parts > 0:
        return args.num_sm_parts
    result = flash_mla_cuda.sparse_decode_fwd(
        tensors["q"],
        tensors["k_cache"],
        tensors["indices"],
        None,  # topk_length
        None,  # attn_sink
        None,  # tile_scheduler_metadata
        None,  # num_splits
        None,  # extra_k_cache
        None,  # extra_indices_in_kvcache
        None,  # extra_topk_length
        args.head_size_v,
        args.softmax_scale,
    )
    sched_meta = result[2]
    if sched_meta is None:
        raise RuntimeError("flash_mla.cuda.sparse_decode_fwd did not return tile_scheduler_metadata.")
    return sched_meta.shape[0]


def _attach_dispatcher_metadata(tensors, args, flash_mla_cuda):
    num_sm_parts = _infer_num_sm_parts(tensors, args, flash_mla_cuda)
    tensors["tile_scheduler_metadata"] = torch.empty((num_sm_parts, 8), device=tensors["q"].device, dtype=torch.int32)
    tensors["num_splits"] = torch.empty((args.batch + 1,), device=tensors["q"].device, dtype=torch.int32)


def _prepare_dispatcher_metadata(core, args):
    _infinicore.deepseek_v4_flashmla_sparse_attention_metadata_(
        core["tile_scheduler_metadata"]._underlying,
        core["num_splits"]._underlying,
        core["topk_length"]._underlying,
        args.topk,
        None,
        -1,
    )


def _run_infinicore(core, args):
    indices = core["indices"]._underlying
    out, lse, _, _ = _infinicore.flash_mla_sparse_decode_fwd(
        core["q"]._underlying,
        core["k_cache"]._underlying,
        indices,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        args.head_size_v,
        args.softmax_scale,
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
    parser = argparse.ArgumentParser(description="Test DeepSeek-V4 FlashMLA sparse_decode_fwd bridge.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--mode", choices=["sparse"], default="sparse")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-q", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=576)
    parser.add_argument("--head-size-v", type=int, default=512)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--cache-seqlen", type=int, default=512)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--num-sm-parts", type=int, default=0)
    parser.add_argument("--bf16-kvcache", action="store_true")
    parser.add_argument("--softmax-scale", type=float, default=1.0 / (576 ** 0.5))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()

    import flash_mla.cuda as flash_mla_cuda

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

    print(f"mode={args.mode}")
    print(
        f"shape q={tuple(tensors['q'].shape)} k_cache={tuple(tensors['k_cache'].shape)} "
        f"indices={tuple(tensors['indices'].shape)} "
        f"topk_length={tuple(tensors['topk_length'].shape)} "
        f"out={tuple(out.shape)} lse={tuple(lse.shape)}"
    )
    print(f"reference avg_ms={ref_avg:.4f} total_ms={ref_total:.4f}")
    print(f"infinicore avg_ms={op_avg:.4f} total_ms={op_total:.4f}")
    print(f"out max_abs={out_abs:.6g} max_rel={out_rel:.6g} equal={out_ok}")
    print(f"lse max_abs={lse_abs:.6g} max_rel={lse_rel:.6g} equal={lse_ok}")

    if not (out_ok and lse_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
