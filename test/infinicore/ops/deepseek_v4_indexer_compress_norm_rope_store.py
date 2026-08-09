import argparse

import infinicore
import torch
from infinicore.lib import _infinicore


HEAD_DIM = 128
ROPE_DIM = 64
VALUE_BYTES_PER_TOKEN = 128
SCALE_BYTES_PER_TOKEN = 4
FP8_MAX = 448.0
SCALE_FACTOR = 0.0022321429569274187
HADAMARD_SCALE = 0.08838834764831845


def _page_bytes(page_size):
    return (VALUE_BYTES_PER_TOKEN + SCALE_BYTES_PER_TOKEN) * page_size


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


def _make_kv(tokens, strided, device):
    if strided:
        base = torch.randn(tokens, HEAD_DIM + 8, device=device, dtype=torch.bfloat16) * 0.2
        return base[:, :HEAD_DIM]
    return (torch.randn(tokens, HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.2).contiguous()


def _make_out_loc(tokens, page_size, dtype, invalid_every, device):
    loc = torch.arange(tokens, device=device, dtype=torch.int64)
    if invalid_every > 0:
        loc[invalid_every - 1 :: invalid_every] = -1
    return loc.to(dtype).contiguous(), loc.cpu()


def _hadamard_rotate(x, bf16_roundtrip):
    y = x.clone()
    span = 1
    while span < HEAD_DIM:
        view = y.reshape(y.shape[0], -1, span * 2)
        a = view[:, :, :span].clone()
        b = view[:, :, span:].clone()
        view[:, :, :span] = a + b
        view[:, :, span:] = a - b
        span *= 2
    y = y * HADAMARD_SCALE
    if bf16_roundtrip:
        y = y.to(torch.bfloat16).float()
    return y


def _reference_cache_python(kv, weight, eps, freqs_cis, positions, out_loc_cpu, blocks, page_size):
    tokens = kv.shape[0]
    ref = torch.zeros((blocks, _page_bytes(page_size)), device=kv.device, dtype=torch.uint8)

    x = kv.float()
    normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight.float()

    tail = normed[:, HEAD_DIM - ROPE_DIM :].reshape(tokens, ROPE_DIM // 2, 2)
    freqs = freqs_cis.index_select(0, positions.long()).reshape(tokens, ROPE_DIM // 2, 2)
    c = freqs[..., 0]
    s = freqs[..., 1]
    xr = tail[..., 0]
    xi = tail[..., 1]
    rope = torch.stack((xr * c - xi * s, xr * s + xi * c), dim=-1).reshape(tokens, ROPE_DIM)
    normed[:, HEAD_DIM - ROPE_DIM :] = rope

    rotated = _hadamard_rotate(normed, bf16_roundtrip=False)
    scale = torch.clamp(rotated.abs().amax(dim=-1, keepdim=True), min=1.0e-4) * SCALE_FACTOR
    quant = torch.clamp(rotated / scale, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8)
    scale_bytes = scale.contiguous().view(torch.uint8).reshape(tokens, SCALE_BYTES_PER_TOKEN)

    flat = ref.reshape(-1)
    for row, loc in enumerate(out_loc_cpu.tolist()):
        if loc < 0:
            continue
        page = loc // page_size
        offset = loc % page_size
        token_base = page * _page_bytes(page_size) + offset * VALUE_BYTES_PER_TOKEN
        scale_base = page * _page_bytes(page_size) + VALUE_BYTES_PER_TOKEN * page_size + offset * SCALE_BYTES_PER_TOKEN
        flat[token_base : token_base + VALUE_BYTES_PER_TOKEN] = quant[row]
        flat[scale_base : scale_base + SCALE_BYTES_PER_TOKEN] = scale_bytes[row]
    return ref


def _reference_cache_kernel(kv, weight, eps, freqs_cis, positions, out_loc, blocks, page_size):
    ref_kv = kv.contiguous().clone()
    ref_cache = torch.zeros((blocks, _page_bytes(page_size)), device=kv.device, dtype=torch.uint8)
    keepalive = []

    _infinicore.deepseek_v4_compress_fused_norm_rope_(
        _as_core(ref_kv, keepalive),
        _as_core(weight, keepalive),
        eps,
        _as_core(freqs_cis, keepalive),
        _as_core(positions, keepalive),
    )
    _infinicore.deepseek_v4_indexer_rotate_(
        _as_core(ref_kv, keepalive),
        True,
    )
    _infinicore.deepseek_v4_store_indexer_raw_cache_(
        _as_core(ref_kv, keepalive),
        _as_core(ref_cache, keepalive),
        _as_core(out_loc, keepalive),
        page_size,
    )
    _sync()
    return ref_cache


def _assert_cache_exact(name, got, ref):
    diff = (got != ref).sum().item()
    if diff != 0:
        raise AssertionError(f"{name}: cache byte mismatch, diff_bytes={diff}")


def _assert_cache_close(name, got, ref, out_loc_cpu, page_size, atol, rtol):
    flat_got = got.reshape(-1)
    flat_ref = ref.reshape(-1)
    got_rows = []
    ref_rows = []
    for loc in out_loc_cpu.tolist():
        if loc < 0:
            continue
        page = loc // page_size
        offset = loc % page_size
        token_base = page * _page_bytes(page_size) + offset * VALUE_BYTES_PER_TOKEN
        scale_base = page * _page_bytes(page_size) + VALUE_BYTES_PER_TOKEN * page_size + offset * SCALE_BYTES_PER_TOKEN
        got_values = flat_got[token_base : token_base + VALUE_BYTES_PER_TOKEN].contiguous().view(torch.float8_e4m3fn).float()
        ref_values = flat_ref[token_base : token_base + VALUE_BYTES_PER_TOKEN].contiguous().view(torch.float8_e4m3fn).float()
        got_scale = flat_got[scale_base : scale_base + SCALE_BYTES_PER_TOKEN].contiguous().view(torch.float32)
        ref_scale = flat_ref[scale_base : scale_base + SCALE_BYTES_PER_TOKEN].contiguous().view(torch.float32)
        got_rows.append(got_values * got_scale)
        ref_rows.append(ref_values * ref_scale)
    if got_rows:
        got_dequant = torch.stack(got_rows, dim=0)
        ref_dequant = torch.stack(ref_rows, dim=0)
        if not torch.allclose(got_dequant, ref_dequant, atol=atol, rtol=rtol):
            max_abs = (got_dequant - ref_dequant).abs().max().item()
            raise AssertionError(f"{name}: cache dequant mismatch, max_abs={max_abs}")


def _run_case(tokens, pos_dtype, out_dtype, weight_dtype, strided, invalid_every, args):
    torch.manual_seed(args.seed + tokens * 17 + (0 if weight_dtype is torch.bfloat16 else 1000))
    device = "cuda"
    kv = _make_kv(tokens, strided, device)
    weight = (torch.randn(HEAD_DIM, device=device, dtype=weight_dtype) * 0.25).contiguous()
    freqs = _make_freqs(args.max_pos, device)
    positions = ((torch.arange(tokens, device=device, dtype=pos_dtype) * 7) % args.max_pos).contiguous()
    out_loc, out_loc_cpu = _make_out_loc(tokens, args.page_size, out_dtype, invalid_every, device)
    blocks = max(1, (tokens + args.page_size - 1) // args.page_size)
    out = torch.zeros((blocks, _page_bytes(args.page_size)), device=device, dtype=torch.uint8)

    keepalive = []
    _infinicore.deepseek_v4_indexer_compress_norm_rope_store_(
        _as_core(kv, keepalive),
        _as_core(weight, keepalive),
        args.eps,
        _as_core(freqs, keepalive),
        _as_core(positions, keepalive),
        _as_core(out_loc, keepalive),
        _as_core(out, keepalive),
        args.page_size,
    )
    _sync()

    ref_kernel = _reference_cache_kernel(kv, weight, args.eps, freqs, positions, out_loc, blocks, args.page_size)
    ref_python = _reference_cache_python(kv, weight, args.eps, freqs, positions, out_loc_cpu, blocks, args.page_size)
    legacy_diff = (out != ref_kernel).sum().item()
    _assert_cache_close(
        f"sglang-v2-python tokens={tokens} pos={pos_dtype} out={out_dtype} weight={weight_dtype} strided={strided}",
        out,
        ref_python,
        out_loc_cpu,
        args.page_size,
        args.atol,
        args.rtol,
    )
    print(
        f"passed tokens={tokens} page_size={args.page_size} pos_dtype={positions.dtype} "
        f"out_dtype={out_loc.dtype} weight_dtype={weight.dtype} strided={strided} "
        f"legacy_diff_bytes={legacy_diff}"
    )


def main():
    parser = argparse.ArgumentParser(description="Test DeepSeek-V4 indexer compress_norm_rope_store InfiniCore op.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--tokens", default="1,2,7,17,64,128")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--max-pos", type=int, default=2048)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--invalid-every", type=int, default=5)
    parser.add_argument("--atol", type=float, default=2.0e-1)
    parser.add_argument("--rtol", type=float, default=2.0e-2)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()

    tokens_list = [int(item) for item in args.tokens.split(",") if item]
    for tokens in tokens_list:
        _run_case(tokens, torch.int32, torch.int32, torch.bfloat16, False, args.invalid_every, args)
        _run_case(tokens, torch.int64, torch.int64, torch.bfloat16, False, args.invalid_every, args)
    _run_case(19, torch.int64, torch.int32, torch.bfloat16, True, args.invalid_every, args)
    _run_case(23, torch.int32, torch.int64, torch.float32, False, args.invalid_every, args)
    print("deepseek_v4_indexer_compress_norm_rope_store: all cases passed")


if __name__ == "__main__":
    main()
