import argparse

import infinicore
import torch
from infinicore.lib import _infinicore


def _core(t):
    return infinicore.from_torch(t)


def _sinkhorn(comb, iters, eps):
    row_max = comb.max(dim=2, keepdim=True).values
    comb = torch.exp(comb - row_max)
    comb = comb / comb.sum(dim=2, keepdim=True) + eps
    comb = comb / (comb.sum(dim=1, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=2, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=1, keepdim=True) + eps)
    return comb


def _mhc_post_ref(x, residual, post, comb):
    return (
        post.unsqueeze(-1) * x.float().unsqueeze(1)
        + torch.einsum("nji,njk->nik", comb, residual.float())
    ).to(x.dtype)


def _rms_norm_ref(x, weight, eps):
    return (x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


def _mhc_pre_ref(
    residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, sinkhorn_repeat
):
    tokens, hc, hidden = residual.shape
    mix_hc = (2 + hc) * hc
    x_flat = residual.reshape(tokens, hc * hidden).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = torch.matmul(x_flat, fn.float().t()) * rsqrt
    pre = torch.sigmoid(mixes[:, :hc] * hc_scale[0].float() + hc_base[:hc].float()) + hc_pre_eps
    post = 2.0 * torch.sigmoid(
        mixes[:, hc : 2 * hc] * hc_scale[1].float() + hc_base[hc : 2 * hc].float()
    )
    comb = mixes[:, 2 * hc : mix_hc].reshape(tokens, hc, hc) * hc_scale[2].float()
    comb = comb + hc_base[2 * hc : mix_hc].float().reshape(1, hc, hc)
    comb = _sinkhorn(comb, sinkhorn_repeat, hc_sinkhorn_eps)
    y = (pre.unsqueeze(-1) * residual.float()).sum(1).to(residual.dtype)
    return y, post, comb


def _run_case(tokens, hc, hidden, dtype):
    torch.manual_seed(20260812 + tokens + hidden)
    rms_eps = 1e-6
    hc_pre_eps = 1e-6
    hc_sinkhorn_eps = 1e-6
    hc_post_mult_value = 2.0
    norm_eps = 1e-6
    sinkhorn_repeat = 5
    mix_hc = (2 + hc) * hc

    x = torch.randn(tokens, hidden, device="cuda", dtype=dtype)
    residual = torch.randn(tokens, hc, hidden, device="cuda", dtype=dtype)
    post_layer_mix = torch.rand(tokens, hc, device="cuda", dtype=torch.float32)
    comb_res_mix = torch.rand(tokens, hc, hc, device="cuda", dtype=torch.float32)
    fn = torch.randn(mix_hc, hc * hidden, device="cuda", dtype=torch.float32) * 0.02
    hc_scale = torch.randn(3, device="cuda", dtype=torch.float32) * 0.1
    hc_base = torch.randn(mix_hc, device="cuda", dtype=torch.float32) * 0.1
    norm_weight = torch.randn(hidden, device="cuda", dtype=dtype) * 0.1

    ref_residual = _mhc_post_ref(x, residual, post_layer_mix, comb_res_mix)
    ref_layer_input, ref_post, ref_comb = _mhc_pre_ref(
        ref_residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, sinkhorn_repeat
    )
    ref_layer_input = _rms_norm_ref(ref_layer_input, norm_weight, norm_eps)

    op_names = ("aten", "kernel", "public") if hc == 4 and hidden == 4096 else ("aten",)
    for name in op_names:
        residual_cur = torch.empty_like(ref_residual)
        post_mix_cur = torch.empty_like(ref_post)
        comb_mix_cur = torch.empty_like(ref_comb)
        layer_input_cur = torch.empty_like(ref_layer_input)
        if name == "public":
            op = _infinicore.deepseek_v4_mhc_fused_post_pre_
        else:
            op = getattr(_infinicore, f"deepseek_v4_mhc_fused_post_pre_{name}_")
        op(
            _core(residual_cur)._underlying,
            _core(post_mix_cur)._underlying,
            _core(comb_mix_cur)._underlying,
            _core(layer_input_cur)._underlying,
            _core(x)._underlying,
            _core(residual)._underlying,
            _core(post_layer_mix)._underlying,
            _core(comb_res_mix)._underlying,
            _core(fn)._underlying,
            _core(hc_scale)._underlying,
            _core(hc_base)._underlying,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            _core(norm_weight)._underlying,
            norm_eps,
        )
        infinicore.sync_stream()
        assert torch.allclose(residual_cur.float(), ref_residual.float(), atol=2e-2, rtol=2e-2), (
            name,
            (residual_cur - ref_residual).abs().max(),
        )
        assert torch.allclose(layer_input_cur.float(), ref_layer_input.float(), atol=2e-2, rtol=2e-2), (
            name,
            (layer_input_cur - ref_layer_input).abs().max(),
        )
        assert torch.allclose(post_mix_cur.float(), ref_post.float(), atol=2e-2, rtol=2e-2), (
            name,
            (post_mix_cur - ref_post).abs().max(),
        )
        assert torch.allclose(comb_mix_cur.float(), ref_comb.float(), atol=2e-2, rtol=2e-2), (
            name,
            (comb_mix_cur - ref_comb).abs().max(),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.parse_args()
    _run_case(3, 4, 64, torch.bfloat16)
    _run_case(7, 4, 128, torch.bfloat16)
    _run_case(1, 4, 4096, torch.bfloat16)
    _run_case(8, 4, 4096, torch.bfloat16)
    print("DeepseekV4MhcFusedPostPre: passed")


if __name__ == "__main__":
    main()
