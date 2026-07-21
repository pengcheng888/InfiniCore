import argparse
import ctypes

import deepgemm
import deepgemm.op
import infinicore
from lightop import op as lightop_op
import lightop.op
import torch


def _as_core(tensor):
    return infinicore.from_torch(tensor)


def _load_vendor_global():
    ctypes.CDLL(lightop.op.__file__, mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(deepgemm.op.__file__, mode=ctypes.RTLD_GLOBAL)


def _topk_values(logits, indices):
    return torch.sort(torch.gather(logits, 1, indices.to(torch.int64)), dim=1, descending=True).values


def _test_prefill():
    torch.manual_seed(0)
    device = "cuda"
    num_q, num_k, heads, head_dim, topk_tokens = 2, 4096, 4, 128, 2048

    q = torch.randn(num_q, heads, head_dim, device=device, dtype=torch.bfloat16) * 0.01
    k = torch.randn(num_k, head_dim, device=device, dtype=torch.bfloat16) * 0.01
    weights = torch.ones((num_q, heads), device=device, dtype=torch.float32)
    cu_seqlen_ks = torch.zeros(num_q, device=device, dtype=torch.int32)
    cu_seqlen_ke = torch.full((num_q,), num_k, device=device, dtype=torch.int32)

    ref_logits = torch.empty((num_q, num_k), device=device, dtype=torch.float32)
    ref_indices = torch.empty((num_q, topk_tokens), device=device, dtype=torch.int32)
    lightop_op.mqa_logits(
        q,
        k,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        num_q,
        num_k,
        heads,
        head_dim,
        None,
        True,
        ref_logits,
    )
    lightop_op.top_k_per_row_prefill(
        ref_logits,
        cu_seqlen_ks,
        cu_seqlen_ke,
        ref_indices,
        num_q,
        ref_logits.stride(0),
        ref_logits.stride(1),
        topk_tokens,
    )
    torch.cuda.synchronize()

    out_logits = torch.empty_like(ref_logits)
    out_indices = torch.empty_like(ref_indices)
    infinicore.deepseek_v4_sparse_attn_indexer_prefill_(
        _as_core(q),
        _as_core(k),
        _as_core(weights),
        _as_core(cu_seqlen_ks),
        _as_core(cu_seqlen_ke),
        _as_core(out_logits),
        _as_core(out_indices),
        None,
        topk_tokens,
        True,
    )
    infinicore.sync_stream()

    assert torch.equal(out_logits, ref_logits)
    assert torch.allclose(_topk_values(ref_logits, out_indices), _topk_values(ref_logits, ref_indices), atol=1e-5)


def _test_decode():
    torch.manual_seed(1)
    device = "cuda"
    batch, next_n, heads, head_dim = 2, 1, 32, 128
    block_kv, max_context_len, topk_tokens, num_sms = 64, 4096, 2048, 80
    num_blocks = batch * (max_context_len // block_kv)

    q = torch.randn(batch, next_n, heads, head_dim, device=device, dtype=torch.bfloat16) * 0.01
    fused_kv_cache = torch.randn(num_blocks, block_kv, 1, head_dim, device=device, dtype=torch.bfloat16) * 0.01
    weights = torch.ones((batch * next_n, heads), device=device, dtype=torch.float32)
    context_lens = torch.tensor([4096, 2048], device=device, dtype=torch.int32)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(batch, -1)

    schedule_meta = deepgemm.get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms)
    ref_logits = deepgemm.paged_mqa_logits(
        q,
        fused_kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        True,
    )
    ref_indices = torch.empty((batch * next_n, topk_tokens), device=device, dtype=torch.int32)
    lightop_op.top_k_per_row_decode(
        ref_logits,
        next_n,
        context_lens,
        ref_indices,
        ref_logits.shape[0],
        ref_logits.stride(0),
        ref_logits.stride(1),
        topk_tokens,
    )
    torch.cuda.synchronize()

    out_logits = torch.empty_like(ref_logits)
    out_indices = torch.empty_like(ref_indices)
    infinicore.deepseek_v4_sparse_attn_indexer_decode_(
        _as_core(q),
        _as_core(fused_kv_cache),
        _as_core(weights),
        _as_core(context_lens),
        _as_core(block_table),
        _as_core(schedule_meta),
        _as_core(out_logits),
        _as_core(out_indices),
        max_context_len,
        next_n,
        topk_tokens,
        True,
    )
    infinicore.sync_stream()

    assert torch.equal(out_logits, ref_logits)
    assert torch.allclose(_topk_values(ref_logits, out_indices), _topk_values(ref_logits, ref_indices), atol=1e-5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.parse_args()
    _load_vendor_global()

    _test_prefill()
    _test_decode()
    print("DeepseekV4SparseAttnIndexer: passed")


if __name__ == "__main__":
    main()
