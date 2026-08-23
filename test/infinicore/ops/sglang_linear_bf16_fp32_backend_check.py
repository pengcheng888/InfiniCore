import argparse
import os
import subprocess
import sys
import traceback


SGLANG_PYTHON = "/workspace_codex/InfiniCore/sglang-0.5.16/python"


def _run_child(backend):
    sys.path.insert(0, SGLANG_PYTHON)

    if backend == "aiter":
        os.environ["SGLANG_USE_AITER"] = "1"
        os.environ["SGLANG_OPT_BF16_FP32_GEMM_ALGO"] = "cublas"
    elif backend == "deep_gemm":
        os.environ["SGLANG_USE_AITER"] = "0"
        os.environ["SGLANG_OPT_BF16_FP32_GEMM_ALGO"] = "deep_gemm"
    elif backend == "deepgemm_direct":
        os.environ["SGLANG_USE_AITER"] = "0"
        os.environ["SGLANG_OPT_BF16_FP32_GEMM_ALGO"] = "cublas"
    elif backend == "torch":
        os.environ["SGLANG_USE_AITER"] = "0"
        os.environ["SGLANG_OPT_BF16_FP32_GEMM_ALGO"] = "cublas"
    else:
        raise ValueError(f"unknown backend: {backend}")

    import torch
    if backend == "deepgemm_direct":
        import deepgemm

        print(f"backend={backend}")
        print(f"deepgemm_file={deepgemm.__file__}")
        print(f"has_bf16_gemm_nt={hasattr(deepgemm, 'bf16_gemm_nt')}")
        print(
            "has_m_grouped_bf16_gemm_nt_contiguous="
            f"{hasattr(deepgemm, 'm_grouped_bf16_gemm_nt_contiguous')}"
        )
        print(f"torch={torch.__version__}")
        print(f"cuda_available={torch.cuda.is_available()}")
        print(f"hip={getattr(torch.version, 'hip', None)}")

        if not torch.cuda.is_available():
            raise RuntimeError("torch cuda device is not available")

        fp32_shapes = [
            (16, 256, 64),
            (32, 256, 4096),
        ]
        bf16_shapes = [
            (16, 256, 64),
            (32, 256, 4096),
            (256, 256, 4096),
        ]

        fp32_linear_ok = True
        for m, n, k in fp32_shapes:
            torch.manual_seed(1000 + m + n + k)
            x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
            y = torch.randn((1, n, k), device="cuda", dtype=torch.bfloat16)
            out = torch.empty((m, n), device="cuda", dtype=torch.float32)
            m_indices = torch.zeros((m,), device="cuda", dtype=torch.int32)
            deepgemm.m_grouped_bf16_gemm_nt_contiguous(x, y, out, m_indices)
            ref = torch.mm(x, y[0].t(), out_dtype=torch.float32)
            torch.cuda.synchronize()
            max_abs = (out.float() - ref.float()).abs().max().item()
            ok = torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
            print(
                f"fp32-out shape m={m} n={n} k={k}: "
                f"out_dtype={out.dtype} max_abs={max_abs:.6g} allclose={ok}"
            )
            if not ok:
                fp32_linear_ok = False

        for m, n, k in bf16_shapes:
            torch.manual_seed(2000 + m + n + k)
            x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
            y = torch.randn((1, n, k), device="cuda", dtype=torch.bfloat16)
            out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            m_indices = torch.zeros((m,), device="cuda", dtype=torch.int32)
            deepgemm.m_grouped_bf16_gemm_nt_contiguous(x, y, out, m_indices)
            ref = torch.mm(x, y[0].t(), out_dtype=torch.float32).to(torch.bfloat16)
            torch.cuda.synchronize()
            max_abs = (out.float() - ref.float()).abs().max().item()
            ok = torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
            print(
                f"bf16-out shape m={m} n={n} k={k}: "
                f"out_dtype={out.dtype} max_abs={max_abs:.6g} allclose={ok}"
            )
            if not ok:
                raise AssertionError(
                    f"{backend} bf16-output mismatch for m={m}, n={n}, k={k}"
                )

        if not fp32_linear_ok:
            raise RuntimeError(
                "deepgemm direct grouped bf16 GEMM is not an fp32-output "
                "linear_bf16_fp32 replacement"
            )

        return 0

    from sglang.jit_kernel.dsv4 import gemm as dsv4_gemm
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.utils import is_hip

    print(f"backend={backend}")
    print(f"sglang_gemm_file={dsv4_gemm.__file__}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"hip={getattr(torch.version, 'hip', None)}")
    print(f"is_hip={is_hip()}")
    print(f"gemm._use_aiter={getattr(dsv4_gemm, '_use_aiter', None)}")
    print(
        "gemm._linear_bf16_fp32_algo="
        f"{getattr(dsv4_gemm, '_linear_bf16_fp32_algo', None)}"
    )
    print(
        "deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM="
        f"{getattr(deep_gemm_wrapper, 'ENABLE_JIT_DEEPGEMM', None)}"
    )

    if backend == "deep_gemm" and not getattr(
        deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", False
    ):
        raise RuntimeError("deep_gemm is not enabled in this environment")

    if not torch.cuda.is_available():
        raise RuntimeError("torch cuda device is not available")

    shapes = [
        (4, 8, 16),
        (32, 256, 4096),
    ]

    for m, n, k in shapes:
        torch.manual_seed(1000 + m + n + k)
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        y = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        out = dsv4_gemm.linear_bf16_fp32(x, y)
        ref = torch.mm(x, y.t(), out_dtype=torch.float32)
        torch.cuda.synchronize()
        max_abs = (out.float() - ref.float()).abs().max().item()
        ok = torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
        print(
            f"shape m={m} n={n} k={k}: "
            f"out_dtype={out.dtype} max_abs={max_abs:.6g} allclose={ok}"
        )
        if not ok:
            raise AssertionError(f"{backend} mismatch for m={m}, n={n}, k={k}")

    return 0


def _run_parent(backends, strict):
    exit_code = 0
    for backend in backends:
        print(f"\n===== {backend} =====")
        env = os.environ.copy()
        env["PYTHONPATH"] = SGLANG_PYTHON + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [sys.executable, __file__, "--child-backend", backend]
        proc = subprocess.run(
            cmd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(proc.stdout, end="")
        if proc.returncode == 0:
            print(f"RESULT {backend}: AVAILABLE")
        else:
            if strict:
                exit_code = 1
            print(f"RESULT {backend}: UNAVAILABLE")
    return exit_code


def main():
    parser = argparse.ArgumentParser()
    backend_choices = ["all", "aiter", "deep_gemm", "deepgemm_direct", "torch"]
    child_backend_choices = ["aiter", "deep_gemm", "deepgemm_direct", "torch"]
    parser.add_argument(
        "--backend",
        choices=backend_choices,
        default="all",
    )
    parser.add_argument("--child-backend", choices=child_backend_choices)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.child_backend:
        try:
            return _run_child(args.child_backend)
        except Exception:
            traceback.print_exc()
            return 1

    backends = (
        ["aiter", "deep_gemm", "deepgemm_direct", "torch"]
        if args.backend == "all"
        else [args.backend]
    )
    return _run_parent(backends, args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
