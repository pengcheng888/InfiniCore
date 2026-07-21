import argparse

import infinicore
import torch
from flash_mla.flash_mla_interface import flash_mla_sparse_fwd


def _as_core(tensor):
    return infinicore.from_torch(tensor)


def test_metadata_object():
    meta, num_splits = infinicore.deepseek_v4_flashmla_metadata_()
    assert hasattr(meta, "have_initialized")
    assert num_splits is None


def test_dense_fp8_metadata():
    cache_seqlens = torch.tensor([64, 96], device="cuda", dtype=torch.int32)
    ref_meta, ref_splits = __import__("flash_mla").get_mla_decoding_metadata_dense_fp8(cache_seqlens, 16, 1)
    got_meta, got_splits = infinicore.deepseek_v4_flashmla_metadata_(
        _as_core(cache_seqlens),
        num_heads_per_head_k=16,
        num_heads_k=1,
        dense_fp8=True,
    )
    assert torch.equal(got_meta, ref_meta)
    assert torch.equal(got_splits, ref_splits)


def test_sparse_prefill_compute():
    torch.manual_seed(4)
    seq_q, seq_kv, heads_q, heads_kv, topk, dim = 2, 128, 2, 1, 128, 576
    q = torch.randn((seq_q, heads_q, dim), device="cuda", dtype=torch.bfloat16)
    kv = torch.randn((seq_kv, heads_kv, dim), device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(topk, device="cuda", dtype=torch.int32).reshape(1, 1, topk)
    indices = indices.repeat(seq_q, heads_kv, 1)
    sm_scale = dim ** -0.5
    ref, ref_max_logits, ref_lse = flash_mla_sparse_fwd(q, kv, indices, sm_scale=sm_scale, d_v=512)
    torch.cuda.synchronize()

    out = torch.empty_like(ref)
    max_logits = torch.empty_like(ref_max_logits)
    lse = torch.empty_like(ref_lse)
    infinicore.deepseek_v4_flashmla_sparse_prefill_(
        _as_core(q),
        _as_core(kv),
        _as_core(indices),
        _as_core(out),
        sm_scale,
        d_v=512,
        max_logits=_as_core(max_logits),
        lse=_as_core(lse),
    )
    infinicore.sync_stream()
    assert torch.equal(out, ref)
    assert torch.equal(max_logits, ref_max_logits)
    assert torch.equal(lse, ref_lse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.parse_args()
    test_metadata_object()
    test_dense_fp8_metadata()
    test_sparse_prefill_compute()
    print("DeepseekV4FlashMLACompute: passed")


if __name__ == "__main__":
    main()
