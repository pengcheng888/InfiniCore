import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
from infinicore.lib import _infinicore
import torch


def _as_core(tensor):
    return infinicore.from_torch(tensor).as_strided(list(tensor.shape), list(tensor.stride()))


def _page_bytes(page_size):
    return ((584 * page_size + 575) // 576) * 576


def _make_freqs(max_pos=2048, dim=64, device="cuda"):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    angles = torch.outer(t, inv_freq)
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1).flatten(-2).contiguous()


def _reference_cache(kv, weight, eps, freqs_cis, positions, out_loc, blocks, page_size):
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
    for row, loc in enumerate(out_loc.cpu().tolist()):
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


def _call(kv, weight, eps, freqs, positions, out_loc, cache, page_size):
    _infinicore.deepseek_v4_fused_k_norm_rope_flashmla_(
        _as_core(kv)._underlying,
        _as_core(weight)._underlying,
        eps,
        _as_core(freqs)._underlying,
        _as_core(positions)._underlying,
        _as_core(out_loc)._underlying,
        _as_core(cache)._underlying,
        page_size,
    )


def _check(tokens, pos_dtype, out_dtype, page_size, strided):
    torch.manual_seed(4400 + tokens + (31 if strided else 0))
    eps = 1.0e-6
    blocks = 3
    if strided:
        storage = torch.randn((tokens, 1536), device="cuda", dtype=torch.bfloat16)
        kv = storage[:, 128:640]
        if tokens > 1:
            assert not kv.is_contiguous()
        assert kv.stride(1) == 1
        assert kv.stride(0) >= 512
    else:
        kv = torch.randn((tokens, 512), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((512,), device="cuda", dtype=torch.bfloat16)
    freqs = _make_freqs(device="cuda")
    positions = (torch.arange(tokens, device="cuda", dtype=pos_dtype) * 3 + 1).contiguous()
    out_loc_cpu = torch.tensor([(i * 37 if i % 5 != 4 else -1) for i in range(tokens)], dtype=torch.int64)
    out_loc = out_loc_cpu.to(device="cuda", dtype=out_dtype)
    cache = torch.zeros((blocks, _page_bytes(page_size)), device="cuda", dtype=torch.uint8)
    ref = _reference_cache(kv, weight, eps, freqs, positions, out_loc_cpu, blocks, page_size)

    _call(kv, weight, eps, freqs, positions, out_loc, cache, page_size)
    infinicore.sync_stream()

    if not torch.equal(cache, ref):
        diff = cache != ref
        raise AssertionError(
            f"cache mismatch tokens={tokens} pos={pos_dtype} out={out_dtype} "
            f"strided={strided} diff_bytes={diff.sum().item()}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    args = parser.parse_args()

    for pos_dtype in (torch.int32, torch.int64):
        for out_dtype in (torch.int32, torch.int64):
            for tokens in (1, 5, 17):
                _check(tokens, pos_dtype, out_dtype, page_size=256, strided=False)
                _check(tokens, pos_dtype, out_dtype, page_size=256, strided=True)
    print("deepseek_v4_fused_k_norm_rope_flashmla ok")


if __name__ == "__main__":
    main()
