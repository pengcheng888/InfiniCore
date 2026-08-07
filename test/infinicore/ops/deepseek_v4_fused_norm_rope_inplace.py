import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
from infinicore.lib import _infinicore
import torch


def _as_core(tensor):
    return infinicore.from_torch(tensor).as_strided(list(tensor.shape), list(tensor.stride()))


def _fused_norm_rope_inplace(kv, weight, eps, freqs, positions):
    _infinicore.deepseek_v4_fused_norm_rope_inplace_(
        _as_core(kv)._underlying,
        _as_core(weight)._underlying,
        eps,
        _as_core(freqs)._underlying,
        _as_core(positions)._underlying,
    )


def _make_freqs(max_pos=2048, dim=64, device="cuda"):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    angles = torch.outer(t, inv_freq)
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1).flatten(-2).contiguous()


def _apply_tail_rope(x, freqs_cis, positions):
    y = x.clone()
    rope_dim = freqs_cis.shape[1]
    tail = y[..., -rope_dim:].float().reshape(*y.shape[:-1], rope_dim // 2, 2)
    freqs = freqs_cis.index_select(0, positions.long()).float().reshape(y.shape[0], rope_dim // 2, 2)
    c = freqs[..., 0]
    s = freqs[..., 1]
    xr = tail[..., 0]
    xi = tail[..., 1]
    y[..., -rope_dim:] = torch.stack((xr * c - xi * s, xr * s + xi * c), dim=-1).reshape_as(y[..., -rope_dim:]).to(y.dtype)
    return y


def _reference_kv(kv, weight, eps, freqs_cis, positions):
    y = kv.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    return _apply_tail_rope(y.to(kv.dtype), freqs_cis, positions)


def _assert_close(name, got, ref):
    if not torch.allclose(got, ref, atol=2e-2, rtol=2e-2):
        diff = (got - ref).abs()
        raise AssertionError(f"{name} mismatch: max={diff.max().item()} mean={diff.float().mean().item()}")


def check_fused_norm_rope_inplace(tokens, pos_dtype, graph):
    torch.manual_seed(2000 + tokens + (0 if pos_dtype is torch.int32 else 100))
    eps = 1e-6
    kv = torch.randn((tokens, 512), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((512,), device="cuda", dtype=torch.bfloat16)
    freqs = _make_freqs(device="cuda")
    positions = torch.arange(tokens, device="cuda", dtype=pos_dtype) * 5
    ref = _reference_kv(kv, weight, eps, freqs, positions)
    got = kv.clone()

    if graph:
        infinicore.start_graph_recording()
        _fused_norm_rope_inplace(got, weight, eps, freqs, positions)
        g = infinicore.stop_graph_recording()
        got.copy_(kv)
        infinicore.sync_stream()
        g.run()
    else:
        _fused_norm_rope_inplace(got, weight, eps, freqs, positions)
    infinicore.sync_stream()
    _assert_close(f"fused_norm_rope_inplace tokens={tokens} graph={graph}", got, ref)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--skip-graph", action="store_true")
    args = parser.parse_args()

    for pos_dtype in (torch.int32, torch.int64):
        for tokens in (1, 5, 17, 128):
            check_fused_norm_rope_inplace(tokens, pos_dtype, graph=False)
            if not args.skip_graph:
                check_fused_norm_rope_inplace(tokens, pos_dtype, graph=True)
    print("deepseek_v4_fused_norm_rope_inplace ok")


if __name__ == "__main__":
    main()
