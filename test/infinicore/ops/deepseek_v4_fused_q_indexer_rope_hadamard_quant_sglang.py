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


def _hadamard_128(x):
    out = x
    span = 1
    while span < HEAD_DIM:
        shaped = out.reshape(*out.shape[:-1], -1, span * 2)
        a = shaped[..., :span].clone()
        b = shaped[..., span:].clone()
        shaped[..., :span] = a + b
        shaped[..., span:] = a - b
        span *= 2
    return out * (HEAD_DIM ** -0.5)


def _run_reference(q, weight, freqs, positions, weight_scale):
    q_ref = q.float()
    q_rope = q_ref[:, :, HEAD_DIM - ROPE_DIM :]
    rope = freqs[positions.long()].view(q.shape[0], 1, ROPE_DIM // 2, 2)
    pairs = q_rope.view(q.shape[0], q.shape[1], ROPE_DIM // 2, 2)
    real = pairs[..., 0].clone()
    imag = pairs[..., 1].clone()
    cos = rope[..., 0]
    sin = rope[..., 1]
    pairs[..., 0] = real * cos - imag * sin
    pairs[..., 1] = real * sin + imag * cos
    q_ref = _hadamard_128(q_ref)
    scale = torch.clamp(q_ref.abs().amax(dim=-1, keepdim=True), min=1.0e-4) / 448.0
    q_fp8 = (q_ref / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    weights_out = weight.float().unsqueeze(-1) * float(weight_scale) * scale
    return q_fp8, weights_out


def _run_infinicore(q, weight, freqs, positions, weight_scale, graph=False):
    q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    weights_out = torch.empty((*weight.shape, 1), device=q.device, dtype=torch.float32)
    keepalive = []
    q_core = _as_core(q, keepalive)
    q_fp8_core = _as_core(q_fp8, keepalive)
    weight_core = _as_core(weight, keepalive)
    weights_out_core = _as_core(weights_out, keepalive)
    freqs_core = _as_core(freqs, keepalive)
    positions_core = _as_core(positions, keepalive)
    if graph:
        infinicore.start_graph_recording()
    _infinicore.deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang_(
        q_core,
        q_fp8_core,
        weight_core,
        weights_out_core,
        weight_scale,
        freqs_core,
        positions_core,
    )
    if graph:
        g = infinicore.stop_graph_recording()
        g.run()
    _sync()
    del keepalive
    return q_fp8, weights_out


def _run_sglang(sglang_op, q, weight, freqs, positions, weight_scale):
    q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    weights_out = torch.empty((*weight.shape, 1), device=q.device, dtype=torch.float32)
    sglang_op(
        q,
        q_fp8,
        weight,
        weights_out,
        float(weight_scale),
        freqs,
        positions,
    )
    torch.cuda.synchronize()
    return q_fp8, weights_out


def _assert_exact(name, got, ref):
    diff = (got.view(torch.uint8) != ref.view(torch.uint8)).sum().item()
    if diff != 0:
        raise AssertionError(f"{name}: byte mismatch diff={diff}")
    return diff


def _byte_diff(got, ref):
    return (got.view(torch.uint8) != ref.view(torch.uint8)).sum().item()


def _assert_byte_diff(name, got, ref, max_diff):
    diff = _byte_diff(got, ref)
    if diff > max_diff:
        raise AssertionError(f"{name}: byte mismatch diff={diff}, max_allowed={max_diff}")
    return diff


def _assert_close(name, got, ref, atol, rtol):
    if not torch.allclose(got, ref, atol=atol, rtol=rtol):
        max_abs = (got - ref).abs().max().item()
        raise AssertionError(f"{name}: max_abs={max_abs}")


def _run_case(tokens, heads, args, graph, sglang_op=None):
    torch.manual_seed(args.seed + tokens * 17)
    device = "cuda"
    q = (torch.randn(tokens, heads, HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.2).contiguous()
    weight = (torch.randn(tokens, heads, device=device, dtype=torch.bfloat16) * 0.25).contiguous()
    freqs = _make_freqs(args.max_pos, device)
    positions = ((torch.arange(tokens, device=device, dtype=torch.int32) * 7) % args.max_pos).contiguous()

    got_q_fp8, got_weights = _run_infinicore(q, weight, freqs, positions, args.weight_scale, graph=graph)
    ref_q_fp8, ref_weights = _run_reference(q, weight, freqs, positions, args.weight_scale)
    python_q_diff = _byte_diff(got_q_fp8, ref_q_fp8)
    if sglang_op is None:
        python_q_diff = _assert_byte_diff("q_fp8_python_ref", got_q_fp8, ref_q_fp8, args.max_python_q_byte_diff)
    _assert_close("weights_python_ref", got_weights, ref_weights, args.atol, args.rtol)

    if sglang_op is not None:
        sgl_q_fp8, sgl_weights = _run_sglang(sglang_op, q, weight, freqs, positions, args.weight_scale)
        _assert_exact("q_fp8_sglang", got_q_fp8, sgl_q_fp8)
        _assert_close("weights_sglang", got_weights, sgl_weights, args.atol, args.rtol)

    print(
        f"passed tokens={tokens} heads={heads} "
        f"graph={graph} "
        f"python_q_byte_diff={python_q_diff} "
        f"weights_max_diff={(got_weights - ref_weights).abs().max().item():.3e}"
    )


def _check_rejects(args):
    device = "cuda"
    q = torch.randn(1, args.heads, HEAD_DIM, device=device, dtype=torch.bfloat16).contiguous()
    weight = torch.randn(1, args.heads, device=device, dtype=torch.bfloat16).contiguous()
    freqs = _make_freqs(args.max_pos, device)
    positions = torch.zeros(1, device=device, dtype=torch.int32)
    q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    weights_out = torch.empty(1, args.heads, 1, device=device, dtype=torch.float32)

    bad_positions = positions.to(torch.int64)
    keepalive = []
    try:
        _infinicore.deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang_(
            _as_core(q, keepalive),
            _as_core(q_fp8, keepalive),
            _as_core(weight, keepalive),
            _as_core(weights_out, keepalive),
            args.weight_scale,
            _as_core(freqs, keepalive),
            _as_core(bad_positions, keepalive),
        )
    except RuntimeError:
        return
    finally:
        del keepalive
    raise AssertionError("int64 positions should be rejected")


def main():
    parser = argparse.ArgumentParser(description="Test SGLang-aligned DeepSeek-V4 fused q indexer rope hadamard quant op.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--tokens", default="1,2,7,16,64,128")
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--max-pos", type=int, default=2048)
    parser.add_argument("--weight-scale", type=float, default=DEFAULT_WEIGHT_SCALE)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--rtol", type=float, default=1.0e-6)
    parser.add_argument("--max-python-q-byte-diff", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--compare-sglang", action="store_true")
    parser.add_argument("--skip-graph", action="store_true")
    args = parser.parse_args()

    sglang_op = None
    if args.compare_sglang:
        try:
            import sgl_kernel

            sglang_op = getattr(sgl_kernel, "dsv4_fused_q_indexer_rope_hadamard_quant", None)
            if sglang_op is None:
                sglang_op = torch.ops.sgl_kernel.dsv4_fused_q_indexer_rope_hadamard_quant
        except (ImportError, AttributeError) as exc:
            print(f"skip sglang compare: {exc}")

    for tokens in _parse_int_list(args.tokens):
        _run_case(tokens, args.heads, args, graph=False, sglang_op=sglang_op)
        if not args.skip_graph:
            _run_case(tokens, args.heads, args, graph=True, sglang_op=sglang_op)
    _check_rejects(args)
    print("deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang: all cases passed")


if __name__ == "__main__":
    main()
