import argparse

import deepgemm
import infinicore
import torch
from deepgemm.m_group_gemm import pack_int8_weight_enk_to_w6_low_latency


def _as_core(tensor):
    return infinicore.from_torch(tensor)


def _routing(m: int, block_m: int):
    sorted_token_ids = torch.arange(m, device="cuda", dtype=torch.int32)
    expert_ids = torch.zeros((m // block_m,), device="cuda", dtype=torch.int32)
    num_tokens_post_pad = torch.tensor([m], device="cuda", dtype=torch.int32)
    topk_weights = torch.ones(m, device="cuda", dtype=torch.float32)
    return sorted_token_ids, expert_ids, num_tokens_post_pad, topk_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.parse_args()

    m, k, n, e = 32, 128, 512, 1
    sorted_token_ids, expert_ids, num_tokens_post_pad, topk_weights = _routing(m, 32)
    input_i8 = torch.randint(-4, 4, (m, k), device="cuda", dtype=torch.int8)
    b_qweight = torch.randint(-4, 4, (e, n, k), device="cuda", dtype=torch.int8)
    a_scale = torch.ones((m, 1), device="cuda", dtype=torch.float32)
    b_scale = torch.ones((e, n, 1), device="cuda", dtype=torch.float32)

    ref_prefill = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    torch.ops.deep_gemm.moe_w8a8_i8_marlin_prefill_down(
        input_i8,
        b_qweight,
        ref_prefill,
        a_scale,
        b_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        1,
        1,
    )
    torch.cuda.synchronize()
    out_prefill = torch.empty_like(ref_prefill)
    infinicore.deepseek_v4_deep_gemm_moe_w8a8_i8_marlin_prefill_down_(
        _as_core(input_i8),
        _as_core(b_qweight),
        _as_core(out_prefill),
        _as_core(a_scale),
        _as_core(b_scale),
        _as_core(topk_weights),
        _as_core(sorted_token_ids),
        _as_core(expert_ids),
        _as_core(num_tokens_post_pad),
        1,
        1,
    )
    infinicore.sync_stream()
    assert torch.equal(out_prefill, ref_prefill)

    ref_decode = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    torch.ops.deep_gemm.moe_w8a8_marlin_decode_down_fp8(
        input_i8,
        b_qweight,
        ref_decode,
        a_scale,
        b_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        1,
        1,
    )
    torch.cuda.synchronize()
    out_decode = torch.empty_like(ref_decode)
    infinicore.deepseek_v4_deep_gemm_moe_w8a8_marlin_decode_down_fp8_(
        _as_core(input_i8),
        _as_core(b_qweight),
        _as_core(out_decode),
        _as_core(a_scale),
        _as_core(b_scale),
        _as_core(topk_weights),
        _as_core(sorted_token_ids),
        _as_core(expert_ids),
        _as_core(num_tokens_post_pad),
        1,
        1,
    )
    infinicore.sync_stream()
    assert torch.equal(out_decode, ref_decode)

    e_ll, m_ll, k_ll, n_ll = 1, 16, 64, 16
    matrix_a = torch.randint(-4, 4, (e_ll, m_ll, k_ll), device="cuda", dtype=torch.int8)
    matrix_b = pack_int8_weight_enk_to_w6_low_latency(
        torch.randint(-4, 4, (e_ll, n_ll, k_ll), device="cuda", dtype=torch.int8)
    )
    matrix_a_scale = torch.ones((e_ll, m_ll), device="cuda", dtype=torch.float32)
    matrix_b_scale = torch.ones((e_ll, n_ll), device="cuda", dtype=torch.float32)
    actual_tokens = torch.tensor([m_ll], device="cuda", dtype=torch.int32)
    ref_ll = torch.empty((e_ll, m_ll, n_ll), device="cuda", dtype=torch.bfloat16)
    torch.ops.deep_gemm.low_latency_grouped_gemm(
        matrix_a,
        matrix_b,
        matrix_a_scale,
        matrix_b_scale,
        actual_tokens,
        ref_ll,
        m_ll,
        e_ll,
        128,
        False,
        False,
        None,
    )
    torch.cuda.synchronize()
    out_ll = torch.empty_like(ref_ll)
    infinicore.deepseek_v4_deep_gemm_low_latency_grouped_gemm_(
        _as_core(matrix_a),
        _as_core(matrix_b),
        _as_core(matrix_a_scale),
        _as_core(matrix_b_scale),
        _as_core(actual_tokens),
        _as_core(out_ll),
        m_ll,
        e_ll,
        128,
        False,
        False,
        None,
    )
    infinicore.sync_stream()
    assert torch.equal(out_ll, ref_ll)

    print("DeepseekV4DeepGEMM: passed")


if __name__ == "__main__":
    main()
