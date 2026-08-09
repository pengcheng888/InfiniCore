import argparse

import infinicore
import torch
from infinicore.lib import _infinicore


HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
VALUE_BYTES_PER_TOKEN = 576
SCALE_BYTES_PER_TOKEN = 8
FP8_MAX = 448.0


def _page_bytes(page_size):
    raw = (VALUE_BYTES_PER_TOKEN + SCALE_BYTES_PER_TOKEN) * page_size
    return ((raw + VALUE_BYTES_PER_TOKEN - 1) // VALUE_BYTES_PER_TOKEN) * VALUE_BYTES_PER_TOKEN


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


def _reference_cache_python(kv, weight, eps, freqs_cis, positions, out_loc_cpu, blocks, page_size):
    tokens = kv.shape[0]
    page_bytes = _page_bytes(page_size)
    ref = torch.zeros((blocks, page_bytes), device=kv.device, dtype=torch.uint8)

    x = kv.float()
    norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight.float()

    nope = norm[:, :NOPE_DIM].to(torch.bfloat16).float().reshape(tokens, 7, 64)
    scale_raw = torch.clamp(nope.abs().amax(dim=-1, keepdim=True), min=1.0e-4) / FP8_MAX
    scale_exp = torch.clamp(torch.ceil(torch.log2(scale_raw)).to(torch.int32) + 127, 0, 255).to(torch.uint8)
    scale = torch.pow(torch.tensor(2.0, device=kv.device), scale_exp.float() - 127.0)
    quant = torch.clamp(nope / scale, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8).reshape(tokens, NOPE_DIM)

    tail = norm[:, NOPE_DIM:].reshape(tokens, 32, 2)
    freqs = freqs_cis.index_select(0, positions.long()).reshape(tokens, 32, 2)
    c = freqs[..., 0]
    s = freqs[..., 1]
    xr = tail[..., 0]
    xi = tail[..., 1]
    rope = torch.stack((xr * c - xi * s, xr * s + xi * c), dim=-1).reshape(tokens, ROPE_DIM)
    rope_bytes = rope.to(torch.bfloat16).contiguous().view(torch.uint8).reshape(tokens, ROPE_DIM * 2)

    flat = ref.reshape(-1)
    for row, loc in enumerate(out_loc_cpu.tolist()):
        if loc < 0:
            continue
        page = loc // page_size
        offset = loc % page_size
        token_base = page * page_bytes + offset * VALUE_BYTES_PER_TOKEN
        scale_base = page * page_bytes + VALUE_BYTES_PER_TOKEN * page_size + offset * SCALE_BYTES_PER_TOKEN
        flat[token_base : token_base + NOPE_DIM] = quant[row]
        flat[token_base + NOPE_DIM : token_base + VALUE_BYTES_PER_TOKEN] = rope_bytes[row]
        flat[scale_base : scale_base + 7] = scale_exp[row].reshape(7)
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
    _infinicore.deepseek_v4_store_flashmla_raw_cache_(
        _as_core(ref_kv, keepalive),
        _as_core(ref_cache, keepalive),
        _as_core(out_loc, keepalive),
        page_size,
    )
    _sync()
    return ref_cache


def _assert_cache_close(
    name,
    got,
    ref,
    out_loc_cpu,
    page_size,
    rope_atol,
    rope_rtol,
    quant_atol=None,
    quant_rtol=None,
):
    page_bytes = _page_bytes(page_size)
    flat_got = got.reshape(-1)
    flat_ref = ref.reshape(-1)
    non_rope_mask = torch.ones_like(flat_ref, dtype=torch.bool)
    quant_mask = torch.zeros_like(flat_ref, dtype=torch.bool)
    got_rope_rows = []
    ref_rope_rows = []
    got_quant_rows = []
    ref_quant_rows = []
    for loc in out_loc_cpu.tolist():
        if loc < 0:
            continue
        page = loc // page_size
        offset = loc % page_size
        token_base = page * page_bytes + offset * VALUE_BYTES_PER_TOKEN
        scale_base = page * page_bytes + VALUE_BYTES_PER_TOKEN * page_size + offset * SCALE_BYTES_PER_TOKEN
        rope_begin = token_base + NOPE_DIM
        rope_end = token_base + VALUE_BYTES_PER_TOKEN
        quant_mask[token_base:rope_begin] = True
        quant_mask[scale_base : scale_base + 7] = True
        non_rope_mask[rope_begin:rope_end] = False
        got_quant_rows.append((flat_got[token_base:rope_begin], flat_got[scale_base : scale_base + 7]))
        ref_quant_rows.append((flat_ref[token_base:rope_begin], flat_ref[scale_base : scale_base + 7]))
        got_rope_rows.append(flat_got[rope_begin:rope_end])
        ref_rope_rows.append(flat_ref[rope_begin:rope_end])

    strict_mask = non_rope_mask & ~quant_mask
    strict_diff = ((flat_got != flat_ref) & strict_mask).sum().item()
    if strict_diff != 0:
        raise AssertionError(f"{name}: cache mismatch outside quant/RoPE regions, diff_bytes={strict_diff}")

    if quant_atol is None:
        quant_diff = ((flat_got != flat_ref) & quant_mask).sum().item()
        if quant_diff != 0:
            raise AssertionError(f"{name}: quant byte mismatch, diff_bytes={quant_diff}")
    elif got_quant_rows:
        got_dequant = []
        ref_dequant = []
        for (got_values, got_scales), (ref_values, ref_scales) in zip(got_quant_rows, ref_quant_rows):
            got_fp8 = got_values.contiguous().view(torch.float8_e4m3fn).float().reshape(7, 64)
            ref_fp8 = ref_values.contiguous().view(torch.float8_e4m3fn).float().reshape(7, 64)
            got_scale = torch.pow(torch.tensor(2.0, device=got.device), got_scales.float().reshape(7, 1) - 127.0)
            ref_scale = torch.pow(torch.tensor(2.0, device=ref.device), ref_scales.float().reshape(7, 1) - 127.0)
            got_dequant.append((got_fp8 * got_scale).reshape(-1))
            ref_dequant.append((ref_fp8 * ref_scale).reshape(-1))
        got_dequant = torch.stack(got_dequant, dim=0)
        ref_dequant = torch.stack(ref_dequant, dim=0)
        if not torch.allclose(got_dequant, ref_dequant, rtol=quant_rtol, atol=quant_atol):
            max_abs = (got_dequant - ref_dequant).abs().max().item()
            raise AssertionError(f"{name}: quant dequant mismatch, max_abs={max_abs}")

    if got_rope_rows:
        got_rope = torch.stack(got_rope_rows, dim=0).contiguous().view(torch.bfloat16).reshape(-1, ROPE_DIM)
        ref_rope = torch.stack(ref_rope_rows, dim=0).contiguous().view(torch.bfloat16).reshape(-1, ROPE_DIM)
        if not torch.allclose(got_rope.float(), ref_rope.float(), rtol=rope_rtol, atol=rope_atol):
            max_abs = (got_rope.float() - ref_rope.float()).abs().max().item()
            raise AssertionError(f"{name}: RoPE BF16 mismatch, max_abs={max_abs}")


def _run_case(tokens, page_size, pos_dtype, out_dtype, strided, invalid_every, args):
    torch.manual_seed(args.seed + tokens * 13 + page_size)
    device = "cuda"
    eps = args.eps
    kv = _make_kv(tokens, strided, device)
    weight = (torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.25).contiguous()
    freqs = _make_freqs(args.max_pos, device)
    positions = ((torch.arange(tokens, device=device, dtype=pos_dtype) * 7) % args.max_pos).contiguous()
    out_loc, out_loc_cpu = _make_out_loc(tokens, page_size, out_dtype, invalid_every, device)
    blocks = max(1, (tokens + page_size - 1) // page_size)
    out = torch.zeros((blocks, _page_bytes(page_size)), device=device, dtype=torch.uint8)

    keepalive = []
    _infinicore.deepseek_v4_compress_norm_rope_store_kernel_(
        _as_core(kv, keepalive),
        _as_core(weight, keepalive),
        eps,
        _as_core(freqs, keepalive),
        _as_core(positions, keepalive),
        _as_core(out_loc, keepalive),
        _as_core(out, keepalive),
        page_size,
    )
    _sync()

    ref_python = _reference_cache_python(kv, weight, eps, freqs, positions, out_loc_cpu, blocks, page_size)
    ref_kernel = _reference_cache_kernel(kv, weight, eps, freqs, positions, out_loc, blocks, page_size)
    _assert_cache_close(
        f"python tokens={tokens} page_size={page_size} pos={pos_dtype} out={out_dtype} strided={strided}",
        out,
        ref_python,
        out_loc_cpu,
        page_size,
        args.rope_atol,
        args.rope_rtol,
    )
    _assert_cache_close(
        f"kernel tokens={tokens} page_size={page_size} pos={pos_dtype} out={out_dtype} strided={strided}",
        out,
        ref_kernel,
        out_loc_cpu,
        page_size,
        args.rope_atol,
        args.rope_rtol,
        # The legacy two-step path stores BF16 after norm/RoPE before quantizing,
        # while the fused op quantizes from FP32 intermediates.
        quant_atol=args.quant_atol,
        quant_rtol=args.quant_rtol,
    )
    _assert_cache_close(
        f"kernel_vs_python tokens={tokens} page_size={page_size} pos={pos_dtype} out={out_dtype} strided={strided}",
        ref_kernel,
        ref_python,
        out_loc_cpu,
        page_size,
        args.rope_atol,
        args.rope_rtol,
        quant_atol=args.quant_atol,
        quant_rtol=args.quant_rtol,
    )
    print(
        f"passed tokens={tokens} page_size={page_size} "
        f"pos_dtype={positions.dtype} out_dtype={out_loc.dtype} strided={strided}"
    )


def main():
    parser = argparse.ArgumentParser(description="Test DeepSeek-V4 compress_norm_rope_store InfiniCore op.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--tokens", default="1,2,7,17,64")
    parser.add_argument("--max-pos", type=int, default=2048)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--invalid-every", type=int, default=5)
    parser.add_argument("--rope-atol", type=float, default=2.0e-2)
    parser.add_argument("--rope-rtol", type=float, default=2.0e-2)
    parser.add_argument("--quant-atol", type=float, default=2.0e-1)
    parser.add_argument("--quant-rtol", type=float, default=2.0e-2)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()

    tokens_list = [int(item) for item in args.tokens.split(",") if item]
    for page_size in (64, 2):
        for tokens in tokens_list:
            _run_case(tokens, page_size, torch.int32, torch.int32, False, args.invalid_every, args)
            _run_case(tokens, page_size, torch.int64, torch.int64, False, args.invalid_every, args)
    _run_case(19, 64, torch.int64, torch.int32, True, args.invalid_every, args)
    print("deepseek_v4_compress_norm_rope_store: all cases passed")


if __name__ == "__main__":
    main()
