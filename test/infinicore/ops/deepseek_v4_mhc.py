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


def _mhc_pre_ref(x, fn, scale, base, rms_eps, hc_eps, sinkhorn_iters):
    tokens, hc, hidden = x.shape
    mix_hc = (2 + hc) * hc
    x_flat = x.reshape(tokens, hc * hidden).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = torch.matmul(x_flat, fn.float().t()) * rsqrt
    pre = torch.sigmoid(mixes[:, :hc] * scale[0].float() + base[:hc].float()) + hc_eps
    post = 2.0 * torch.sigmoid(
        mixes[:, hc : 2 * hc] * scale[1].float() + base[hc : 2 * hc].float()
    )
    comb = mixes[:, 2 * hc : mix_hc].reshape(tokens, hc, hc) * scale[2].float()
    comb = comb + base[2 * hc : mix_hc].float().reshape(1, hc, hc)
    comb = _sinkhorn(comb, sinkhorn_iters, hc_eps)
    y = (pre.unsqueeze(-1) * x.float()).sum(1).to(x.dtype)
    return y, post, comb


def _mhc_post_ref(x, residual, post, comb):
    return (
        post.unsqueeze(-1) * x.float().unsqueeze(1)
        + torch.einsum("nji,njk->nik", comb, residual.float())
    ).to(x.dtype)


def _mhc_head_ref(x, fn, scale, base, rms_eps, hc_eps):
    tokens, hc, hidden = x.shape
    x_flat = x.reshape(tokens, hc * hidden).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = torch.matmul(x_flat, fn.float().t()) * rsqrt
    pre = torch.sigmoid(mixes * scale[0].float() + base.float()) + hc_eps
    return (pre.unsqueeze(-1) * x.float()).sum(1).to(x.dtype)


def _run_case(tokens, hc, hidden, dtype):
    torch.manual_seed(20260721 + tokens + hidden)
    rms_eps = 1e-6
    hc_eps = 1e-6
    sinkhorn_iters = 5
    mix_hc = (2 + hc) * hc
    x = torch.randn(tokens, hc, hidden, device="cuda", dtype=dtype)
    fn = torch.randn(mix_hc, hc * hidden, device="cuda", dtype=torch.float32) * 0.02
    scale = torch.randn(3, device="cuda", dtype=torch.float32) * 0.1
    base = torch.randn(mix_hc, device="cuda", dtype=torch.float32) * 0.1

    ref_y, ref_post, ref_comb = _mhc_pre_ref(
        x, fn, scale, base, rms_eps, hc_eps, sinkhorn_iters
    )
    y = torch.empty_like(ref_y)
    post = torch.empty_like(ref_post)
    comb = torch.empty_like(ref_comb)
    _infinicore.deepseek_v4_mhc_pre_naive_(
        _core(y)._underlying,
        _core(post)._underlying,
        _core(comb)._underlying,
        _core(x)._underlying,
        _core(fn)._underlying,
        _core(scale)._underlying,
        _core(base)._underlying,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
    )
    infinicore.sync_stream()
    assert torch.allclose(y.float(), ref_y.float(), atol=2e-2, rtol=2e-2), (
        y - ref_y
    ).abs().max()
    assert torch.allclose(post.float(), ref_post.float(), atol=2e-2, rtol=2e-2), (
        post - ref_post
    ).abs().max()
    assert torch.allclose(comb.float(), ref_comb.float(), atol=2e-2, rtol=2e-2), (
        comb - ref_comb
    ).abs().max()

    post_out_ref = _mhc_post_ref(ref_y, x, ref_post, ref_comb)
    post_out = torch.empty_like(post_out_ref)
    _infinicore.deepseek_v4_mhc_post_naive_(
        _core(post_out)._underlying,
        _core(y)._underlying,
        _core(x)._underlying,
        _core(post)._underlying,
        _core(comb)._underlying,
    )
    infinicore.sync_stream()
    assert torch.allclose(
        post_out.float(), post_out_ref.float(), atol=2e-2, rtol=2e-2
    ), (post_out - post_out_ref).abs().max()

    head_fn = torch.randn(hc, hc * hidden, device="cuda", dtype=torch.float32) * 0.02
    head_scale = torch.randn(1, device="cuda", dtype=torch.float32) * 0.1
    head_base = torch.randn(hc, device="cuda", dtype=torch.float32) * 0.1
    head_ref = _mhc_head_ref(post_out, head_fn, head_scale, head_base, rms_eps, hc_eps)
    head = torch.empty_like(head_ref)
    _infinicore.deepseek_v4_mhc_head_naive_(
        _core(head)._underlying,
        _core(post_out)._underlying,
        _core(head_fn)._underlying,
        _core(head_scale)._underlying,
        _core(head_base)._underlying,
        rms_eps,
        hc_eps,
    )
    infinicore.sync_stream()
    assert torch.allclose(head.float(), head_ref.float(), atol=2e-2, rtol=2e-2), (
        head - head_ref
    ).abs().max()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.parse_args()
    _run_case(3, 4, 64, torch.bfloat16)
    _run_case(7, 4, 128, torch.bfloat16)
    print("DeepseekV4MHC: passed")


if __name__ == "__main__":
    main()
