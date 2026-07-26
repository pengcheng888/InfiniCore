import argparse

import infinicore
import torch


def _as_core(tensor):
    return infinicore.from_torch(tensor)


def _run_case(m, n, k):
    torch.manual_seed(12 + m + n + k)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    ref = x.float() @ weight.float().t()

    out_default = torch.empty((m, n), device="cuda", dtype=torch.float32)
    infinicore.deepseek_v4_linear_bf16_fp32(_as_core(x), _as_core(weight), out=_as_core(out_default))
    infinicore.sync_stream()
    assert torch.allclose(out_default, ref, atol=2e-2, rtol=2e-2)

    out_naive = torch.empty((m, n), device="cuda", dtype=torch.float32)
    infinicore.deepseek_v4_linear_bf16_fp32(_as_core(x), _as_core(weight), out=_as_core(out_naive))
    infinicore.sync_stream()
    assert torch.allclose(out_naive, ref, atol=1e-4, rtol=1e-4)

    out_kernel = torch.empty((m, n), device="cuda", dtype=torch.float32)
    infinicore.deepseek_v4_linear_bf16_fp32(_as_core(x), _as_core(weight), out=_as_core(out_kernel))
    infinicore.sync_stream()
    assert torch.allclose(out_kernel, out_naive, atol=2e-2, rtol=2e-2)

    ret = infinicore.deepseek_v4_linear_bf16_fp32(_as_core(x), _as_core(weight))
    infinicore.sync_stream()
    assert ret.shape == list(ref.shape)
    assert ret.dtype == infinicore.float32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.parse_args()
    _run_case(4, 8, 16)
    _run_case(7, 11, 32)
    print("DeepseekV4LinearBf16Fp32: passed")


if __name__ == "__main__":
    main()
