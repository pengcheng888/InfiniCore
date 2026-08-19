import argparse

import infinicore
import torch
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor as CoreTensor

DEFAULT_TOKENS = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"
DEFAULT_HEAD_DIM = 512
DEFAULT_GROUPS_C4 = 2
DEFAULT_GROUPS_C128 = 1


def _as_core(tensor):
    return infinicore.from_torch(tensor)._underlying


def _sync():
    infinicore.sync_stream()
    torch.cuda.synchronize()


def _copy_to_torch(core_tensor, shape, dtype=torch.bfloat16):
    out = torch.empty(shape, device="cuda", dtype=dtype)
    infinicore.from_torch(out).copy_(CoreTensor(core_tensor))
    _sync()
    return out


def _assert_close(name, got, ref, atol=2e-2, rtol=2e-2):
    try:
        torch.testing.assert_close(got, ref, atol=atol, rtol=rtol)
    except AssertionError as exc:
        diff = (got.float() - ref.float()).abs().max().item()
        raise AssertionError(f"{name} mismatch, max_abs={diff}") from exc


def _parse_tokens(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


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


def check_c4_sglang_stateful(ape_shape):
    torch.manual_seed(3 + ape_shape[0])
    tokens, dim = 8, 512
    kv_score = torch.randn((tokens, 4 * dim), device="cuda", dtype=torch.bfloat16)
    ape = torch.randn(ape_shape, device="cuda", dtype=torch.bfloat16)
    state_base = torch.zeros((8, 4 * dim), device="cuda", dtype=torch.float32)
    write_loc = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device="cuda", dtype=torch.int32)
    extra_loc = torch.tensor([-1, -1, -1, -1, 0, 0, 0, 0], device="cuda", dtype=torch.int32).reshape(tokens, 1)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int32)

    ref_state = state_base.clone()
    got_state = state_base.clone()
    ref_core = _infinicore.deepseek_v4_c4_compress_stateful_naive(
        _as_core(kv_score),
        _as_core(ape),
        _as_core(ref_state),
        _as_core(write_loc),
        _as_core(extra_loc),
        _as_core(positions),
    )
    got_core = _infinicore.deepseek_v4_c4_compress_sglang_stateful_kernel(
        _as_core(kv_score),
        _as_core(ape),
        _as_core(got_state),
        _as_core(write_loc),
        _as_core(extra_loc),
        _as_core(positions),
    )
    ref = _copy_to_torch(ref_core, (tokens, dim), kv_score.dtype)
    got = _copy_to_torch(got_core, (tokens, dim), kv_score.dtype)
    _assert_close(f"c4 sglang output ape={ape_shape}", got, ref)
    _assert_close(f"c4 sglang state ape={ape_shape}", got_state, ref_state, atol=0, rtol=0)


def check_c4_rejects_legacy_ape_shape():
    torch.manual_seed(13)
    tokens, dim = 8, 512
    kv_score = torch.randn((tokens, 4 * dim), device="cuda", dtype=torch.bfloat16)
    ape = torch.randn((4, 2 * dim), device="cuda", dtype=torch.bfloat16)
    state = torch.zeros((8, 4 * dim), device="cuda", dtype=torch.float32)
    write_loc = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device="cuda", dtype=torch.int32)
    extra_loc = torch.tensor([-1, -1, -1, -1, 0, 0, 0, 0], device="cuda", dtype=torch.int32).reshape(tokens, 1)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int32)

    try:
        _infinicore.deepseek_v4_c4_compress_sglang_stateful_kernel(
            _as_core(kv_score),
            _as_core(ape),
            _as_core(state),
            _as_core(write_loc),
            _as_core(extra_loc),
            _as_core(positions),
        )
    except RuntimeError as exc:
        if "expects ape [8, head_dim]" not in str(exc):
            raise AssertionError(f"unexpected legacy ape error: {exc}") from exc
        return
    raise AssertionError("c4 sglang kernel accepted legacy ape [4, 2 * head_dim]")


def check_c128_sglang_stateful():
    torch.manual_seed(4)
    tokens, dim = 128, 512
    kv_score = torch.randn((tokens, 2 * dim), device="cuda", dtype=torch.bfloat16)
    ape = torch.randn((128, dim), device="cuda", dtype=torch.bfloat16)
    state_base = torch.zeros((128, 2 * dim), device="cuda", dtype=torch.float32)
    write_loc = torch.zeros((tokens,), device="cuda", dtype=torch.int32)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int32)

    ref_state = state_base.clone()
    got_state = state_base.clone()
    ref_core = _infinicore.deepseek_v4_c128_compress_stateful_naive(
        _as_core(kv_score), _as_core(ape), _as_core(ref_state), _as_core(write_loc), _as_core(positions)
    )
    got_core = _infinicore.deepseek_v4_c128_compress_sglang_stateful_kernel(
        _as_core(kv_score), _as_core(ape), _as_core(got_state), _as_core(write_loc), _as_core(positions)
    )
    ref = _copy_to_torch(ref_core, (tokens, dim), kv_score.dtype)
    got = _copy_to_torch(got_core, (tokens, dim), kv_score.dtype)
    _assert_close("c128 sglang output", got, ref)
    _assert_close("c128 sglang state", got_state, ref_state, atol=0, rtol=0)


def _make_c4_case(tokens, args):
    torch.manual_seed(args.seed + tokens * 17 + 4)
    dim = args.head_dim
    groups = max(args.c4_groups, (tokens + 3) // 4)
    kv_score = torch.randn((tokens, 4 * dim), device="cuda", dtype=torch.bfloat16)
    ape = torch.randn((8, dim), device="cuda", dtype=torch.bfloat16)
    state = torch.randn((4 * groups, 4 * dim), device="cuda", dtype=torch.float32)
    write_loc = (torch.arange(tokens, device="cuda", dtype=torch.int32) // 4).contiguous()
    extra_loc = torch.clamp(write_loc - 1, min=-1).reshape(tokens, 1).contiguous()
    positions = torch.arange(tokens, device="cuda", dtype=torch.int32)
    return kv_score, ape, state, write_loc, extra_loc, positions


def _make_c128_case(tokens, args):
    torch.manual_seed(args.seed + tokens * 17 + 128)
    dim = args.head_dim
    groups = max(args.c128_groups, (tokens + 127) // 128)
    kv_score = torch.randn((tokens, 2 * dim), device="cuda", dtype=torch.bfloat16)
    ape = torch.randn((128, dim), device="cuda", dtype=torch.bfloat16)
    state = torch.randn((128 * groups, 2 * dim), device="cuda", dtype=torch.float32)
    write_loc = (torch.arange(tokens, device="cuda", dtype=torch.int32) // 128).contiguous()
    positions = torch.arange(tokens, device="cuda", dtype=torch.int32)
    return kv_score, ape, state, write_loc, positions


def _run_c4_case(tokens, args):
    kv_score, ape, state, write_loc, extra_loc, positions = _make_c4_case(tokens, args)
    dim = args.head_dim

    def ref_fn():
        return _infinicore.deepseek_v4_c4_compress_stateful_naive(
            _as_core(kv_score), _as_core(ape), _as_core(state.clone()), _as_core(write_loc), _as_core(extra_loc), _as_core(positions)
        )

    def op_fn():
        return _infinicore.deepseek_v4_c4_compress_sglang_stateful(
            _as_core(kv_score), _as_core(ape), _as_core(state.clone()), _as_core(write_loc), _as_core(extra_loc), _as_core(positions)
        )

    ref_once = _copy_to_torch(ref_fn(), (tokens, dim), kv_score.dtype)
    got_once = _copy_to_torch(op_fn(), (tokens, dim), kv_score.dtype)
    max_abs = (got_once.float() - ref_once.float()).abs().max().item()
    allclose = torch.allclose(got_once.float(), ref_once.float(), atol=args.atol, rtol=args.rtol)
    if not allclose:
        raise AssertionError(f"c4 tokens={tokens} mismatch, max_abs={max_abs}")

    ref = _bench(ref_fn, args.warmup, args.iters)
    op = _bench(op_fn, args.warmup, args.iters)
    return "c4", tokens, ref["avg_ms"], op["avg_ms"], max_abs, allclose


def _run_c128_case(tokens, args):
    kv_score, ape, state, write_loc, positions = _make_c128_case(tokens, args)
    dim = args.head_dim

    def ref_fn():
        return _infinicore.deepseek_v4_c128_compress_stateful_naive(
            _as_core(kv_score), _as_core(ape), _as_core(state.clone()), _as_core(write_loc), _as_core(positions)
        )

    def op_fn():
        return _infinicore.deepseek_v4_c128_compress_sglang_stateful(
            _as_core(kv_score), _as_core(ape), _as_core(state.clone()), _as_core(write_loc), _as_core(positions)
        )

    ref_once = _copy_to_torch(ref_fn(), (tokens, dim), kv_score.dtype)
    got_once = _copy_to_torch(op_fn(), (tokens, dim), kv_score.dtype)
    max_abs = (got_once.float() - ref_once.float()).abs().max().item()
    allclose = torch.allclose(got_once.float(), ref_once.float(), atol=args.atol, rtol=args.rtol)
    if not allclose:
        raise AssertionError(f"c128 tokens={tokens} mismatch, max_abs={max_abs}")

    iters = max(1, args.iters // max(1, args.c128_iter_divisor))
    ref = _bench(ref_fn, args.warmup, iters)
    op = _bench(op_fn, args.warmup, iters)
    return "c128", tokens, ref["avg_ms"], op["avg_ms"], max_abs, allclose


def _print_header(args):
    print(f"head_dim={args.head_dim} tokens={args.tokens} iters={args.iters} warmup={args.warmup}")
    print(f"{'case':>5} | {'tokens':>8} | {'naive avg':>10} | {'op avg':>10} | {'speedup':>7} | {'max_abs':>10} | {'allclose':>8}")
    print("-" * 83)


def _print_row(result):
    case, tokens, ref_avg, op_avg, max_abs, allclose = result
    speedup = ref_avg / op_avg if op_avg > 0 else float("inf")
    print(f"{case:>5} | {tokens:8d} | {ref_avg:10.4f} | {op_avg:10.4f} | {speedup:7.2f} | {max_abs:10.3e} | {str(allclose):>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--c4-groups", type=int, default=DEFAULT_GROUPS_C4)
    parser.add_argument("--c128-groups", type=int, default=DEFAULT_GROUPS_C128)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--c128-iter-divisor", type=int, default=4)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    check_c4_sglang_stateful((8, 512))
    check_c4_rejects_legacy_ape_shape()
    check_c128_sglang_stateful()
    _print_header(args)
    for tokens in _parse_tokens(args.tokens):
        _print_row(_run_c4_case(tokens, args))
    for tokens in _parse_tokens(args.tokens):
        _print_row(_run_c128_case(tokens, args))
    print("deepseek_v4_compress_sglang_stateful: all cases passed")


if __name__ == "__main__":
    main()
