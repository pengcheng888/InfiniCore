import argparse

import infinicore
import torch
from infinicore.lib import _infinicore


def _as_core(tensor):
    return infinicore.from_torch(tensor)._underlying


def _run_case(m, n, k, use_bias):
    torch.manual_seed(2026 + m * 13 + n * 17 + k)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).contiguous()
    weight = torch.randint(-128, 127, (n, k), device="cuda", dtype=torch.int8).contiguous()
    weight_scale = (torch.rand((1, n), device="cuda", dtype=torch.float32) * 0.02 + 0.001).contiguous()
    bias = None
    if use_bias:
        bias = (torch.randn((n,), device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()

    out_ref = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    out_workspace = torch.empty_like(out_ref)
    q_input = torch.empty((m, k), device="cuda", dtype=torch.int8)
    input_scale = torch.empty((m, 1), device="cuda", dtype=torch.float32)

    _infinicore.linear_w8a8i8_(
        _as_core(out_ref),
        _as_core(x),
        _as_core(weight),
        _as_core(weight_scale),
        None if bias is None else _as_core(bias),
    )
    _infinicore.linear_w8a8i8_out_workspace_(
        _as_core(out_workspace),
        _as_core(x),
        _as_core(weight),
        _as_core(weight_scale),
        None if bias is None else _as_core(bias),
        _as_core(q_input),
        _as_core(input_scale),
    )
    infinicore.sync_stream()
    torch.cuda.synchronize()

    if not torch.equal(out_ref, out_workspace):
        max_diff = (out_ref.float() - out_workspace.float()).abs().max().item()
        raise AssertionError(f"m={m} n={n} k={k} bias={use_bias} max_diff={max_diff}")
    if q_input.dtype != torch.int8 or input_scale.dtype != torch.float32:
        raise AssertionError("workspace dtype mismatch")
    print(f"m={m:<4d} n={n:<5d} k={k:<5d} bias={int(use_bias)} ok")


def _run_lmslim_compare_case(m, n, k):
    torch.manual_seed(4096 + m)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).contiguous()
    weight = torch.randint(-64, 64, (n, k), device="cuda", dtype=torch.int8).contiguous()
    weight_scale_native = (torch.rand((1, n), device="cuda", dtype=torch.float32) * 0.02 + 0.001).contiguous()
    weight_scale_lmslim = weight_scale_native.t().contiguous()
    smooth_scale = torch.ones((k,), device="cuda", dtype=torch.float32).contiguous()

    native_out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    lmslim_out = torch.empty_like(native_out)
    native_q = torch.empty((m, k), device="cuda", dtype=torch.int8)
    native_scale = torch.empty((m, 1), device="cuda", dtype=torch.float32)
    lmslim_q = torch.empty_like(native_q)
    lmslim_scale = torch.empty_like(native_scale)

    _infinicore.linear_w8a8i8_out_workspace_(
        _as_core(native_out),
        _as_core(x),
        _as_core(weight),
        _as_core(weight_scale_native),
        None,
        _as_core(native_q),
        _as_core(native_scale),
    )
    _infinicore.deepseek_v4_lmslim_linear_w8a8_(
        _as_core(lmslim_out),
        _as_core(x),
        _as_core(weight).permute([1, 0]),
        _as_core(weight_scale_lmslim),
        None,
        _as_core(lmslim_q),
        _as_core(lmslim_scale),
        _as_core(smooth_scale),
    )
    infinicore.sync_stream()
    torch.cuda.synchronize()

    torch.testing.assert_close(native_q.float(), lmslim_q.float(), atol=1, rtol=0)
    torch.testing.assert_close(native_scale, lmslim_scale, atol=1.0e-6, rtol=1.0e-6)
    diff = (native_out.float() - lmslim_out.float()).abs()
    close = torch.isclose(native_out.float(), lmslim_out.float(), atol=1.5e-1, rtol=5.0e-2)
    mismatch = close.numel() - int(close.sum().item())
    if mismatch > close.numel() // 100:
        raise AssertionError(
            f"native_vs_lmslim mismatch too large: {mismatch}/{close.numel()} max_diff={diff.max().item()}")
    print(
        f"native_vs_lmslim m={m:<4d} n={n:<5d} k={k:<5d} "
        f"max_diff={diff.max().item():.6f} mean_diff={diff.mean().item():.6f} mismatch={mismatch}/{close.numel()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    args = parser.parse_args()
    del args

    _run_case(1, 32, 64, False)
    _run_case(17, 1536, 4096, False)
    _run_case(64, 1536, 4096, False)
    _run_case(32, 1536, 4096, True)
    _run_lmslim_compare_case(17, 1536, 4096)
    _run_lmslim_compare_case(64, 1536, 4096)
    print("linear_w8a8i8_workspace: passed")


if __name__ == "__main__":
    main()
