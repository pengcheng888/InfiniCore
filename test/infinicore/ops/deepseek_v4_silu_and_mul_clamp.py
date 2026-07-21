import argparse

import infinicore
import torch


def _as_core(tensor):
    return infinicore.from_torch(tensor)


def _reference(x, limit):
    gate, up = x.chunk(2, dim=-1)
    gate = torch.minimum(gate, torch.tensor(float(limit), device=gate.device, dtype=gate.dtype))
    up = torch.clamp(up, min=-float(limit), max=float(limit))
    gate_f = gate.float()
    return ((gate_f / (1.0 + torch.exp(-gate_f))) * up.float()).to(x.dtype)


def _run_case(shape, dtype, limit):
    torch.manual_seed(11)
    x = torch.randn(shape, device="cuda", dtype=dtype) * 4
    ref = _reference(x, limit)
    out = torch.empty_like(ref)
    got = infinicore.deepseek_v4_silu_and_mul_clamp(_as_core(x), limit, out=_as_core(out))
    assert got is not None
    infinicore.sync_stream()
    assert torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)

    ret = infinicore.deepseek_v4_silu_and_mul_clamp(_as_core(x), limit)
    infinicore.sync_stream()
    assert ret.shape == list(ref.shape)
    assert ret.dtype == infinicore.from_torch(ref).dtype


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.parse_args()
    _run_case((4, 512), torch.bfloat16, 7.0)
    _run_case((2, 3, 1024), torch.float16, 5.5)
    print("DeepseekV4SiluAndMulClamp: passed")


if __name__ == "__main__":
    main()
