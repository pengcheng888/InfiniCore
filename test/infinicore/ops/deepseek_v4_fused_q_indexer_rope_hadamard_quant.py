import argparse

import infinicore
import torch
from infinicore.lib import _infinicore


HEAD_DIM = 128
ROPE_DIM = 64
DEFAULT_HEADS = 64
DEFAULT_WEIGHT_SCALE = (HEAD_DIM ** -0.5) * (DEFAULT_HEADS ** -0.5)


def _parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _as_core(tensor, keepalive):
    base = infinicore.from_torch(tensor)
    wrapped = base.as_strided(list(tensor.shape), list(tensor.stride()))
    keepalive.append(base)
    keepalive.append(wrapped)
    return wrapped._underlying


def _sync():
    infinicore.sync_stream()
    torch.cuda.synchronize()


def _make_freqs(max_pos, device):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, ROPE_DIM, 2, device=device, dtype=torch.float32) / ROPE_DIM))
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    angles = torch.outer(t, inv_freq)
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1).flatten(-2).contiguous()


def _run_baseline(q, weights, freqs, positions, weight_scale):
    q_ref = q.clone()
    q_rope = q_ref[:, :, HEAD_DIM - ROPE_DIM :]
    q_fp8 = torch.empty_like(q_ref, dtype=torch.float8_e4m3fn)
    q_scale = torch.empty((*q_ref.shape[:-1], 1), device=q_ref.device, dtype=torch.float32)
    fused_weights = torch.empty_like(weights, dtype=torch.float32)
    keepalive = []
    _infinicore.deepseek_v4_fused_rope_(
        _as_core(q_rope, keepalive),
        None,
        _as_core(freqs, keepalive),
        _as_core(positions, keepalive),
        False,
    )
    _infinicore.deepseek_v4_indexer_rotate_(
        _as_core(q_ref, keepalive),
        True,
    )
    _infinicore.deepseek_v4_c4_act_quant_fused_scale_kernel_(
        _as_core(q_ref, keepalive),
        _as_core(weights, keepalive),
        _as_core(q_fp8, keepalive),
        _as_core(q_scale, keepalive),
        _as_core(fused_weights, keepalive),
        weight_scale,
    )
    _sync()
    del keepalive
    return q_fp8, q_scale, fused_weights


def _run_fused(q, weights, freqs, positions, weight_scale):
    q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    q_scale = torch.empty((*q.shape[:-1], 1), device=q.device, dtype=torch.float32)
    fused_weights = torch.empty_like(weights, dtype=torch.float32)
    keepalive = []
    _infinicore.deepseek_v4_fused_q_indexer_rope_hadamard_quant_(
        _as_core(q, keepalive),
        _as_core(weights, keepalive),
        _as_core(q_fp8, keepalive),
        _as_core(q_scale, keepalive),
        _as_core(fused_weights, keepalive),
        weight_scale,
        _as_core(freqs, keepalive),
        _as_core(positions, keepalive),
    )
    _sync()
    del keepalive
    return q_fp8, q_scale, fused_weights


def _assert_exact(name, got, ref):
    diff = (got.view(torch.uint8) != ref.view(torch.uint8)).sum().item()
    if diff != 0:
        raise AssertionError(f"{name}: byte mismatch diff={diff}")


def _assert_close(name, got, ref, atol, rtol):
    if not torch.allclose(got, ref, atol=atol, rtol=rtol):
        max_abs = (got - ref).abs().max().item()
        raise AssertionError(f"{name}: max_abs={max_abs}")


def _run_case(tokens, heads, weight_dtype, pos_dtype, args):
    torch.manual_seed(args.seed + tokens * 17 + (0 if weight_dtype is torch.bfloat16 else 1000))
    device = "cuda"
    q = (torch.randn(tokens, heads, HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.2).contiguous()
    weights = (torch.randn(tokens, heads, device=device, dtype=weight_dtype) * 0.25).contiguous()
    freqs = _make_freqs(args.max_pos, device)
    positions = ((torch.arange(tokens, device=device, dtype=pos_dtype) * 7) % args.max_pos).contiguous()

    ref_q_fp8, ref_q_scale, ref_fused_weights = _run_baseline(q, weights, freqs, positions, args.weight_scale)
    got_q_fp8, got_q_scale, got_fused_weights = _run_fused(q, weights, freqs, positions, args.weight_scale)

    _assert_exact("q_fp8", got_q_fp8, ref_q_fp8)
    _assert_close("q_scale", got_q_scale, ref_q_scale, args.atol, args.rtol)
    _assert_close("fused_weights", got_fused_weights, ref_fused_weights, args.atol, args.rtol)
    print(
        f"passed tokens={tokens} heads={heads} weight_dtype={weights.dtype} "
        f"pos_dtype={positions.dtype} q_scale_max_diff={(got_q_scale - ref_q_scale).abs().max().item():.3e} "
        f"weights_max_diff={(got_fused_weights - ref_fused_weights).abs().max().item():.3e}"
    )


def main():
    parser = argparse.ArgumentParser(description="Test DeepSeek-V4 fused q indexer rope hadamard quant op.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--tokens", default="1,2,7,16,64,128")
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--max-pos", type=int, default=2048)
    parser.add_argument("--weight-scale", type=float, default=DEFAULT_WEIGHT_SCALE)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    for tokens in _parse_int_list(args.tokens):
        _run_case(tokens, args.heads, torch.bfloat16, torch.int32, args)
        _run_case(tokens, args.heads, torch.bfloat16, torch.int64, args)
        _run_case(tokens, args.heads, torch.float32, torch.int32, args)
    print("deepseek_v4_fused_q_indexer_rope_hadamard_quant: all cases passed")


if __name__ == "__main__":
    main()
