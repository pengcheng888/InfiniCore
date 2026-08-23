import argparse

import torch
from lightop import op as lightop_op

from infinicore._preload import preload_device


def _assert_equal(name, out, ref):
    if not torch.equal(out, ref):
        max_diff = (out.float() - ref.float()).abs().max().item()
        raise AssertionError(f"{name} mismatch: max_diff={max_diff}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.parse_args()

    torch.manual_seed(0)
    preload_device("HYGON")

    m, k, n, e = 32, 128, 512, 1
    input_i8 = torch.randint(-4, 4, (m, k), device="cuda", dtype=torch.int8)
    b_qweight = torch.randint(-4, 4, (e, n, k), device="cuda", dtype=torch.int8)
    a_scale = torch.ones((m, 1), device="cuda", dtype=torch.float32)
    b_scale = torch.ones((e, n, 1), device="cuda", dtype=torch.float32)
    sorted_token_ids = torch.arange(m, device="cuda", dtype=torch.int32)
    expert_ids = torch.zeros((m // 16,), device="cuda", dtype=torch.int32)
    num_tokens_post_pad = torch.tensor([m], device="cuda", dtype=torch.int32)
    topk_weights = torch.ones(m, device="cuda", dtype=torch.float32)

    ref_gemm = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    lightop_op.moe_gemm_marlin_w8a8(
        input_i8, b_qweight, ref_gemm, a_scale, b_scale, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_pad, 1, 0, 0)
    torch.cuda.synchronize()

    out_gemm = torch.empty_like(ref_gemm)
    torch.ops.infinicore_deepseek_v4.lightop_moe_gemm_marlin_w8a8(
        input_i8, b_qweight, out_gemm, a_scale, b_scale, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_pad, 1, 0, 0)
    torch.cuda.synchronize()
    _assert_equal("lightop_moe_gemm_marlin_w8a8", out_gemm, ref_gemm)

    x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    ref_q = torch.empty((64, 256), device="cuda", dtype=torch.int8)
    ref_scale = torch.empty((64, 1), device="cuda", dtype=torch.float32)
    lightop_op.fuse_silu_mul_quant(x, ref_q, ref_scale, None, 1, -1, None)
    torch.cuda.synchronize()

    out_q = torch.empty_like(ref_q)
    out_scale = torch.empty_like(ref_scale)
    torch.ops.infinicore_deepseek_v4.lightop_fuse_silu_mul_quant(
        x, out_q, out_scale, None, 1, -1, None)
    torch.cuda.synchronize()
    _assert_equal("lightop_fuse_silu_mul_quant.output", out_q, ref_q)
    _assert_equal("lightop_fuse_silu_mul_quant.scale", out_scale, ref_scale)

    down = torch.randn(32, 8, 128, device="cuda", dtype=torch.bfloat16)
    ref_sum = torch.empty((32, 128), device="cuda", dtype=torch.bfloat16)
    lightop_op.moe_sum(down, ref_sum, None, None, None, 1.0, -1)
    torch.cuda.synchronize()

    out_sum = torch.empty_like(ref_sum)
    torch.ops.infinicore_deepseek_v4.lightop_moe_sum(
        down, out_sum, None, None, None, 1.0, -1)
    torch.cuda.synchronize()
    _assert_equal("lightop_moe_sum", out_sum, ref_sum)

    print("DeepseekV4LightopMoeMarlin: passed")


if __name__ == "__main__":
    main()
