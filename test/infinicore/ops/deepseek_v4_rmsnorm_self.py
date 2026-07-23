import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch


def _as_core(tensor):
    return infinicore.from_torch(tensor)


def _reference(x, eps):
    y = x.float()
    return (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def run_case(shape, dtype, eps):
    torch.manual_seed(len(shape) + shape[-1])
    x = torch.randn(shape, device="cuda", dtype=dtype)
    ref = _reference(x, eps)
    out = torch.empty_like(x)
    infinicore.deepseek_v4_rmsnorm_self_(_as_core(out), _as_core(x), eps)
    infinicore.sync_stream()
    assert torch.allclose(out, ref, atol=2e-2, rtol=2e-2), (out - ref).abs().max()

    out2 = torch.empty_like(x)
    infinicore.deepseek_v4_rmsnorm_self(_as_core(x), eps, out=_as_core(out2))
    infinicore.sync_stream()
    assert torch.allclose(out2, ref, atol=2e-2, rtol=2e-2), (out2 - ref).abs().max()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.parse_args()
    for shape in ((1, 8, 512), (7, 8, 512), (11, 64)):
        run_case(shape, torch.bfloat16, 1e-6)
    print("deepseek_v4_rmsnorm_self ok")


if __name__ == "__main__":
    main()
