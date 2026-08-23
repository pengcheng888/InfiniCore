import os

os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")

import argparse

import infinicore
import torch


FP8_E4M3_MAX = 448.0


def _as_core(tensor):
    return infinicore.from_torch(tensor)


def _fp8_empty(shape):
    return torch.empty(shape, device="cuda", dtype=torch.float8_e4m3fnuz)


def _assert_fp8_equal(actual, expected):
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def _silu_and_mul_ref(x, swiglu_limit=None):
    gate, up = x.chunk(2, dim=-1)
    if swiglu_limit is not None:
        gate = torch.minimum(gate, torch.tensor(float(swiglu_limit), device=gate.device, dtype=gate.dtype))
        up = torch.clamp(up, min=-float(swiglu_limit), max=float(swiglu_limit))
    gate_f = gate.float()
    return (gate_f / (1.0 + torch.exp(-gate_f))) * up.float()


def _fp8_group_quant_ref(values, group_size, dtype):
    groups = values.reshape(*values.shape[:-1], values.shape[-1] // group_size, group_size)
    absmax = torch.clamp(groups.abs().amax(dim=-1), min=1.0e-10)
    scale = absmax / FP8_E4M3_MAX
    quant = torch.clamp(groups / scale.unsqueeze(-1), min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX)
    return quant.reshape(values.shape).to(dtype), scale.float()


def _ue8m0_group_quant_ref(values, group_size, dtype):
    groups = values.reshape(values.shape[0], values.shape[-1] // group_size, group_size).float()
    raw_scale = torch.clamp(groups.abs().amax(dim=-1), min=1.0e-10) / FP8_E4M3_MAX
    exp = torch.ceil(torch.log2(raw_scale)).to(torch.int32) + 127
    exp = torch.clamp(exp, min=0, max=255)
    scale = torch.pow(2.0, exp.float() - 127.0)
    quant = torch.clamp(groups / scale.unsqueeze(-1), min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX)
    return quant.reshape(values.shape).to(dtype), exp.to(torch.uint8)


def test_silu_and_mul_quant_contig():
    torch.manual_seed(0)
    m, hidden = 4, 512
    x = torch.randn((m, hidden * 2), device="cuda", dtype=torch.bfloat16)
    ref_out, ref_scale = _fp8_group_quant_ref(_silu_and_mul_ref(x), 128, torch.float8_e4m3fnuz)

    out = _fp8_empty((m, hidden))
    scale = torch.empty((m, hidden // 128), device="cuda", dtype=torch.float32)
    infinicore.deepseek_v4_silu_and_mul_quant_(
        _as_core(x),
        _as_core(out),
        _as_core(scale),
        None,
        quant_group_size=128,
    )
    infinicore.sync_stream()
    _assert_fp8_equal(out, ref_out)
    assert torch.equal(scale, ref_scale)


def test_silu_and_mul_quant_masked():
    torch.manual_seed(1)
    experts, tokens, hidden, topk = 4, 8, 512, 3
    x = torch.randn((experts, tokens, hidden * 2), device="cuda", dtype=torch.bfloat16)
    masked_m = torch.tensor([2, 1, 3, 0], device="cuda", dtype=torch.int32)
    values = _silu_and_mul_ref(x)
    ref_out = _fp8_empty((experts, tokens, hidden))
    ref_scale = torch.empty((experts, tokens, hidden // 128), device="cuda", dtype=torch.float32)
    ref_out.zero_()
    ref_scale.zero_()
    for expert_id, count in enumerate(masked_m.cpu().tolist()):
        valid = int(count)
        if valid <= 0:
            continue
        quant, scales = _fp8_group_quant_ref(values[expert_id, :valid], 128, ref_out.dtype)
        ref_out[expert_id, :valid].copy_(quant)
        ref_scale[expert_id, :valid].copy_(scales)

    out = _fp8_empty((experts, tokens, hidden))
    scale = torch.empty_like(ref_scale)
    out.zero_()
    scale.zero_()
    infinicore.deepseek_v4_silu_and_mul_quant_(
        _as_core(x),
        _as_core(out),
        _as_core(scale),
        _as_core(masked_m),
        quant_group_size=128,
        topk=topk,
    )
    infinicore.sync_stream()
    for expert_id, count in enumerate(masked_m.cpu().tolist()):
        valid = int(count)
        if valid <= 0:
            continue
        _assert_fp8_equal(out[expert_id, :valid], ref_out[expert_id, :valid])
        assert torch.equal(scale[expert_id, :valid], ref_scale[expert_id, :valid])


def test_mega_moe_pre_dispatch():
    torch.manual_seed(2)
    m, hidden, padded, topk, group_size = 5, 128, 8, 3, 32
    x = torch.randn((m, hidden), device="cuda", dtype=torch.bfloat16)
    topk_idx = torch.randint(0, 8, (m, topk), device="cuda", dtype=torch.int32)
    topk_weights = torch.rand((m, topk), device="cuda", dtype=torch.float32)
    ref_x, ref_exp = _ue8m0_group_quant_ref(x, group_size, torch.float8_e4m3fnuz)

    out_x = _fp8_empty((padded, hidden))
    out_x_sf = torch.empty((padded, hidden // group_size // 4), device="cuda", dtype=torch.int32)
    out_topk_idx = torch.empty((padded, topk), device="cuda", dtype=torch.int64)
    out_topk_weights = torch.empty((padded, topk), device="cuda", dtype=torch.float32)
    infinicore.deepseek_v4_mega_moe_pre_dispatch_(
        _as_core(x),
        _as_core(topk_idx),
        _as_core(topk_weights),
        _as_core(out_x),
        _as_core(out_x_sf),
        _as_core(out_topk_idx),
        _as_core(out_topk_weights),
        group_size,
    )
    infinicore.sync_stream()
    _assert_fp8_equal(out_x[:m], ref_x)
    assert torch.equal(out_x_sf.view(torch.uint8).reshape(padded, -1)[:m, : ref_exp.shape[-1]], ref_exp)
    assert torch.equal(out_topk_idx[:m], topk_idx.to(out_topk_idx.dtype))
    assert torch.equal(out_topk_weights[:m], topk_weights)
    assert torch.equal(out_topk_idx[m:], torch.full_like(out_topk_idx[m:], -1))
    assert torch.equal(out_topk_weights[m:], torch.zeros_like(out_topk_weights[m:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.parse_args()
    test_silu_and_mul_quant_contig()
    test_silu_and_mul_quant_masked()
    test_mega_moe_pre_dispatch()
    print("DeepseekV4MoEPreDispatch: passed")


if __name__ == "__main__":
    main()
