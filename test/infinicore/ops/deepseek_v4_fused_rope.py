import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch


def _as_core(tensor):
    return infinicore.from_torch(tensor)


def _reference_apply(x, freqs_cis, positions, inverse=False):
    y = x.clone()
    batch = y.shape[0]
    pair_shape = (*y.shape[:-1], y.shape[-1] // 2, 2)
    x_pair = y.float().reshape(pair_shape)
    freqs = freqs_cis.index_select(0, positions.long()).float().reshape(batch, y.shape[-1] // 2, 2)
    freq_real = freqs[..., 0]
    freq_imag = freqs[..., 1]
    if y.ndim == 3:
        freq_real = freq_real.unsqueeze(1)
        freq_imag = freq_imag.unsqueeze(1)
    x_real = x_pair[..., 0]
    x_imag = x_pair[..., 1]
    if inverse:
        out_real = x_real * freq_real + x_imag * freq_imag
        out_imag = x_imag * freq_real - x_real * freq_imag
    else:
        out_real = x_real * freq_real - x_imag * freq_imag
        out_imag = x_real * freq_imag + x_imag * freq_real
    return torch.stack((out_real, out_imag), dim=-1).reshape_as(y).to(y.dtype)


def _make_freqs(max_pos=128, dim=64, device="cuda"):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    angles = torch.outer(t, inv_freq)
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1).flatten(-2).contiguous()


def run_case(inverse):
    torch.manual_seed(11 if not inverse else 12)
    device = "cuda"
    tokens, heads, rope_dim = 7, 8, 64
    q = torch.randn((tokens, heads, rope_dim), device=device, dtype=torch.bfloat16)
    k = torch.randn((tokens, 1, rope_dim), device=device, dtype=torch.bfloat16)
    positions = torch.tensor([0, 3, 5, 9, 17, 33, 65], device=device, dtype=torch.int64)
    freqs = _make_freqs(device=device)

    q_ref = _reference_apply(q, freqs, positions, inverse)
    k_ref = _reference_apply(k, freqs, positions, inverse)
    q_out = q.clone()
    k_out = k.clone()
    infinicore.deepseek_v4_fused_rope_(
        _as_core(q_out),
        _as_core(k_out),
        _as_core(freqs),
        _as_core(positions),
        inverse,
    )
    infinicore.sync_stream()
    assert torch.allclose(q_out, q_ref, atol=1e-2, rtol=1e-2), (q_out - q_ref).abs().max()
    assert torch.allclose(k_out, k_ref, atol=1e-2, rtol=1e-2), (k_out - k_ref).abs().max()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.parse_args()
    run_case(False)
    run_case(True)
    print("deepseek_v4_fused_rope ok")


if __name__ == "__main__":
    main()
