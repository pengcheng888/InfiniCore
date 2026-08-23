import argparse

import aiter
import infinicore
import torch


def _as_core(tensor):
    return infinicore.from_torch(tensor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.parse_args()

    m, k, n, e = 32, 128, 512, 1
    input_i8 = torch.randint(-4, 4, (m, k), device="cuda", dtype=torch.int8)
    b_qweight = torch.randint(-4, 4, (e, n, k), device="cuda", dtype=torch.int8)
    a_scale = torch.ones((m, 1), device="cuda", dtype=torch.float32)
    b_scale = torch.ones((e, n, 1), device="cuda", dtype=torch.float32)
    sorted_token_ids = torch.arange(m, device="cuda", dtype=torch.int32)
    expert_ids = torch.zeros((m // 16,), device="cuda", dtype=torch.int32)
    num_tokens_post_pad = torch.tensor([m], device="cuda", dtype=torch.int32)
    topk_weights = torch.ones(m, device="cuda", dtype=torch.float32)

    ref = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    torch.ops.aiter.moe_c_moe_gemm_marlin_w8a8(
        input_i8,
        b_qweight,
        ref,
        a_scale,
        b_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        1,
        0,
        0,
    )
    torch.cuda.synchronize()
    out = torch.empty_like(ref)
    infinicore.deepseek_v4_moe_marlin_w8a8_(
        _as_core(input_i8),
        _as_core(b_qweight),
        _as_core(out),
        _as_core(a_scale),
        _as_core(b_scale),
        _as_core(topk_weights),
        _as_core(sorted_token_ids),
        _as_core(expert_ids),
        _as_core(num_tokens_post_pad),
        1,
        0,
        0,
    )
    infinicore.sync_stream()
    assert torch.equal(out, ref)

    ref_fp8 = torch.empty_like(ref)
    torch.ops.aiter.moe_c_moe_gemm_marlin_w8a8_fp8(
        input_i8,
        b_qweight,
        ref_fp8,
        a_scale,
        b_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        1,
        0,
        0,
    )
    torch.cuda.synchronize()
    out_fp8 = torch.empty_like(ref_fp8)
    infinicore.deepseek_v4_moe_marlin_w8a8_fp8_(
        _as_core(input_i8),
        _as_core(b_qweight),
        _as_core(out_fp8),
        _as_core(a_scale),
        _as_core(b_scale),
        _as_core(topk_weights),
        _as_core(sorted_token_ids),
        _as_core(expert_ids),
        _as_core(num_tokens_post_pad),
        1,
        0,
        0,
    )
    infinicore.sync_stream()
    assert torch.isfinite(out_fp8.float()).all()
    assert torch.allclose(out_fp8.float(), ref_fp8.float(), atol=1e-6, rtol=1e-6)

    print("DeepseekV4MoeMarlinW8A8: passed")


if __name__ == "__main__":
    main()
