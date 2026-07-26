import argparse
import time

import infinicore
import torch
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor as CoreTensor


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


def _freqs(max_pos=512):
    torch.manual_seed(100)
    theta = torch.randn((max_pos, 32), device="cuda", dtype=torch.float32) * 0.03
    return torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1).reshape(max_pos, 64).contiguous()


def _assert_close(name, got, ref, atol=2e-2, rtol=2e-2):
    try:
        torch.testing.assert_close(got, ref, atol=atol, rtol=rtol)
    except AssertionError as exc:
        diff = (got.float() - ref.float()).abs().max().item()
        raise AssertionError(f"{name} mismatch, max_abs={diff}") from exc


def _bench(label, fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    _sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    ms = (time.perf_counter() - start) * 1000.0 / iters
    print(f"{label:52s} {ms:9.4f} ms")
    return ms


def check_rmsnorm():
    torch.manual_seed(1)
    eps = 1e-6
    x = torch.randn((17, 512), device="cuda", dtype=torch.bfloat16)
    ref_core = _infinicore.deepseek_v4_rmsnorm_self_naive(_as_core(x), eps)
    got_core = _infinicore.deepseek_v4_rmsnorm_self_kernel(_as_core(x), eps)
    ref = _copy_to_torch(ref_core, x.shape, x.dtype)
    got = _copy_to_torch(got_core, x.shape, x.dtype)
    _assert_close("rmsnorm_self return", got, ref)

    ref_out = torch.empty_like(x)
    got_out = torch.empty_like(x)
    _infinicore.deepseek_v4_rmsnorm_self_naive_(_as_core(ref_out), _as_core(x), eps)
    _infinicore.deepseek_v4_rmsnorm_self_kernel_(_as_core(got_out), _as_core(x), eps)
    _sync()
    _assert_close("rmsnorm_self out", got_out, ref_out)


def check_compress_fused_norm_rope():
    torch.manual_seed(2)
    eps = 1e-6
    tokens, dim = 11, 512
    base = torch.randn((tokens, dim), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((dim,), device="cuda", dtype=torch.bfloat16)
    freqs = _freqs(256)
    positions = torch.tensor([0, 1, 2, 3, 7, 11, 31, 63, 127, 128, 129], device="cuda", dtype=torch.int32)

    ref = base.clone()
    got = base.clone()
    _infinicore.deepseek_v4_compress_fused_norm_rope_naive_(
        _as_core(ref), _as_core(weight), eps, _as_core(freqs), _as_core(positions)
    )
    _infinicore.deepseek_v4_compress_fused_norm_rope_kernel_(
        _as_core(got), _as_core(weight), eps, _as_core(freqs), _as_core(positions)
    )
    _sync()
    _assert_close("compress_fused_norm_rope", got, ref)


def check_c4_stateful(ape_shape):
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
    got_core = _infinicore.deepseek_v4_c4_compress_stateful_kernel(
        _as_core(kv_score),
        _as_core(ape),
        _as_core(got_state),
        _as_core(write_loc),
        _as_core(extra_loc),
        _as_core(positions),
    )
    ref = _copy_to_torch(ref_core, (tokens, dim), kv_score.dtype)
    got = _copy_to_torch(got_core, (tokens, dim), kv_score.dtype)
    _assert_close(f"c4 output ape={ape_shape}", got, ref)
    _assert_close(f"c4 state ape={ape_shape}", got_state, ref_state, atol=0, rtol=0)


def check_c128_stateful():
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
    got_core = _infinicore.deepseek_v4_c128_compress_stateful_kernel(
        _as_core(kv_score), _as_core(ape), _as_core(got_state), _as_core(write_loc), _as_core(positions)
    )
    ref = _copy_to_torch(ref_core, (tokens, dim), kv_score.dtype)
    got = _copy_to_torch(got_core, (tokens, dim), kv_score.dtype)
    _assert_close("c128 output", got, ref)
    _assert_close("c128 state", got_state, ref_state, atol=0, rtol=0)


def benchmark(iters):
    print("\nATen naive vs native kernel average latency")

    eps = 1e-6
    x = torch.randn((1, 512), device="cuda", dtype=torch.bfloat16)
    _bench(
        "deepseek_v4_rmsnorm_self_naive",
        lambda: _infinicore.deepseek_v4_rmsnorm_self_naive(_as_core(x), eps),
        iters,
    )
    _bench(
        "deepseek_v4_rmsnorm_self_kernel",
        lambda: _infinicore.deepseek_v4_rmsnorm_self_kernel(_as_core(x), eps),
        iters,
    )

    fused = torch.randn((1, 512), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((512,), device="cuda", dtype=torch.bfloat16)
    freqs = _freqs(512)
    positions = torch.tensor([127], device="cuda", dtype=torch.int32)
    _bench(
        "deepseek_v4_compress_fused_norm_rope_",
        lambda: _infinicore.deepseek_v4_compress_fused_norm_rope_naive_(
            _as_core(fused), _as_core(weight), eps, _as_core(freqs), _as_core(positions)
        ),
        iters,
    )
    _bench(
        "deepseek_v4_compress_fused_norm_rope_",
        lambda: _infinicore.deepseek_v4_compress_fused_norm_rope_kernel_(
            _as_core(fused), _as_core(weight), eps, _as_core(freqs), _as_core(positions)
        ),
        iters,
    )

    kv_c4 = torch.randn((1, 2048), device="cuda", dtype=torch.bfloat16)
    ape_c4 = torch.randn((8, 512), device="cuda", dtype=torch.bfloat16)
    state_c4_ref = torch.randn((8, 2048), device="cuda", dtype=torch.float32)
    state_c4_kernel = state_c4_ref.clone()
    write_c4 = torch.tensor([1], device="cuda", dtype=torch.int32)
    extra_c4 = torch.tensor([[0]], device="cuda", dtype=torch.int32)
    pos_c4 = torch.tensor([7], device="cuda", dtype=torch.int32)
    _bench(
        "deepseek_v4_c4_compress_stateful",
        lambda: _infinicore.deepseek_v4_c4_compress_stateful_naive(
            _as_core(kv_c4), _as_core(ape_c4), _as_core(state_c4_ref), _as_core(write_c4), _as_core(extra_c4), _as_core(pos_c4)
        ),
        iters,
    )
    _bench(
        "deepseek_v4_c4_compress_stateful",
        lambda: _infinicore.deepseek_v4_c4_compress_stateful_kernel(
            _as_core(kv_c4), _as_core(ape_c4), _as_core(state_c4_kernel), _as_core(write_c4), _as_core(extra_c4), _as_core(pos_c4)
        ),
        iters,
    )

    kv_c128 = torch.randn((1, 1024), device="cuda", dtype=torch.bfloat16)
    ape_c128 = torch.randn((128, 512), device="cuda", dtype=torch.bfloat16)
    state_c128_ref = torch.randn((128, 1024), device="cuda", dtype=torch.float32)
    state_c128_kernel = state_c128_ref.clone()
    write_c128 = torch.tensor([0], device="cuda", dtype=torch.int32)
    pos_c128 = torch.tensor([127], device="cuda", dtype=torch.int32)
    c128_iters = max(10, iters // 4)
    _bench(
        "deepseek_v4_c128_compress_stateful",
        lambda: _infinicore.deepseek_v4_c128_compress_stateful_naive(
            _as_core(kv_c128), _as_core(ape_c128), _as_core(state_c128_ref), _as_core(write_c128), _as_core(pos_c128)
        ),
        c128_iters,
    )
    _bench(
        "deepseek_v4_c128_compress_stateful",
        lambda: _infinicore.deepseek_v4_c128_compress_stateful_kernel(
            _as_core(kv_c128), _as_core(ape_c128), _as_core(state_c128_kernel), _as_core(write_c128), _as_core(pos_c128)
        ),
        c128_iters,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    check_rmsnorm()
    check_compress_fused_norm_rope()
    check_c4_stateful((8, 512))
    check_c4_stateful((4, 1024))
    check_c128_stateful()
    print("deepseek_v4_attention_compress_native_kernels correctness ok")
    benchmark(args.iters)


if __name__ == "__main__":
    main()
