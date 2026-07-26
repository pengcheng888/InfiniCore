import argparse

import infinicore
import flash_mla.cuda as flash_mla_cuda
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


def test_sparse_decode_attention_compute():
    torch.manual_seed(11)
    page_size = 256
    bytes_per_token = 584
    page_bytes = ((bytes_per_token * page_size + 575) // 576) * 576
    blocks, cache_tokens, tokens, heads, topk = 2, 128, 3, 4, 64

    raw = torch.zeros((blocks, page_bytes), device="cuda", dtype=torch.uint8)
    kv = torch.randn((cache_tokens, 512), device="cuda", dtype=torch.bfloat16)
    slot = torch.arange(cache_tokens, device="cuda", dtype=torch.int32)
    infinicore.deepseek_v4_store_flashmla_raw_cache_(
        _as_core(kv),
        _as_core(raw),
        _as_core(slot),
        page_size,
    )
    infinicore.sync_stream()

    q = torch.randn((tokens, heads, 512), device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(topk, device="cuda", dtype=torch.int32).reshape(1, topk).repeat(tokens, 1)
    topk_lengths = torch.full((tokens,), topk, device="cuda", dtype=torch.int32)
    attn_sink = torch.zeros((heads,), device="cuda", dtype=torch.float32)

    k_cache = raw[:, : page_size * bytes_per_token].view(torch.float8_e4m3fn).view(
        blocks, page_size, 1, bytes_per_token
    )
    ref, _, _, _ = flash_mla_cuda.sparse_decode_fwd(
        q.unsqueeze(1),
        k_cache,
        indices.unsqueeze(1),
        topk_lengths,
        attn_sink,
        None,
        None,
        None,
        None,
        None,
        512,
        512**-0.5,
    )
    torch.cuda.synchronize()

    out = torch.empty_like(q)
    infinicore.deepseek_v4_flashmla_sparse_attention_(
        _as_core(q),
        _as_core(raw),
        _as_core(indices),
        _as_core(topk_lengths),
        _as_core(attn_sink),
        _as_core(out),
        512**-0.5,
        page_size,
        512,
    )
    infinicore.sync_stream()
    assert torch.equal(out, ref.squeeze(1))


def _flashmla_page_bytes(page_size):
    bytes_per_token = 584
    value_alignment = 576
    return ((bytes_per_token * page_size + value_alignment - 1) // value_alignment) * value_alignment


def _make_flashmla_raw_cache(page_size, blocks, cache_tokens, seed):
    torch.manual_seed(seed)
    raw = torch.zeros((blocks, _flashmla_page_bytes(page_size)), device="cuda", dtype=torch.uint8)
    kv = torch.randn((cache_tokens, 512), device="cuda", dtype=torch.bfloat16)
    slot = torch.arange(cache_tokens, device="cuda", dtype=torch.int32)
    infinicore.deepseek_v4_store_flashmla_raw_cache_(
        _as_core(kv),
        _as_core(raw),
        _as_core(slot),
        page_size,
    )
    infinicore.sync_stream()
    return raw


def _run_sparse_attention(raw, q, indices, topk_lengths, attn_sink, extra=None):
    out = torch.empty_like(q)
    kwargs = {}
    if extra is not None:
        extra_raw, extra_indices, extra_topk_lengths, extra_page_size = extra
        kwargs.update(
            extra_raw_cache=_as_core(extra_raw),
            extra_indices=_as_core(extra_indices),
            extra_topk_lengths=_as_core(extra_topk_lengths),
            extra_page_size=extra_page_size,
        )
    infinicore.deepseek_v4_flashmla_sparse_attention_(
        _as_core(q),
        _as_core(raw),
        _as_core(indices),
        _as_core(topk_lengths),
        _as_core(attn_sink),
        _as_core(out),
        512**-0.5,
        256,
        512,
        **kwargs,
    )
    infinicore.sync_stream()
    return out


def _assert_local_heads_match_full_heads(use_extra):
    torch.manual_seed(23 if not use_extra else 29)
    tokens = 3
    global_heads = 16
    tp_size = 8
    local_heads = global_heads // tp_size
    topk = 64
    page_size = 256

    raw = _make_flashmla_raw_cache(page_size, blocks=2, cache_tokens=128, seed=31)
    q_full = torch.randn((tokens, global_heads, 512), device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(topk, device="cuda", dtype=torch.int32).reshape(1, topk).repeat(tokens, 1)
    topk_lengths = torch.full((tokens,), topk, device="cuda", dtype=torch.int32)
    attn_sink = torch.linspace(-0.25, 0.25, global_heads, device="cuda", dtype=torch.float32)

    extra = None
    if use_extra:
        extra_page_size = 64
        extra_topk = 32
        extra_raw = _make_flashmla_raw_cache(extra_page_size, blocks=2, cache_tokens=96, seed=37)
        extra_indices = torch.arange(extra_topk, device="cuda", dtype=torch.int32).reshape(1, extra_topk).repeat(tokens, 1)
        extra_topk_lengths = torch.full((tokens,), extra_topk, device="cuda", dtype=torch.int32)
        extra = (extra_raw, extra_indices, extra_topk_lengths, extra_page_size)

    full_out = _run_sparse_attention(raw, q_full, indices, topk_lengths, attn_sink, extra=extra)
    for tp_rank in (0, 3, 7):
        head_start = tp_rank * local_heads
        head_stop = head_start + local_heads
        q_local = q_full[:, head_start:head_stop, :].contiguous()
        sink_local = attn_sink[head_start:head_stop].contiguous()
        local_out = _run_sparse_attention(raw, q_local, indices, topk_lengths, sink_local, extra=extra)
        ref = full_out[:, head_start:head_stop, :].contiguous()
        assert torch.equal(local_out, ref), f"local-head FlashMLA mismatch at tp_rank={tp_rank}, use_extra={use_extra}"


def test_sparse_decode_local_heads_match_full_heads():
    _assert_local_heads_match_full_heads(use_extra=False)
    _assert_local_heads_match_full_heads(use_extra=True)


def test_sparse_decode_attention_cached_metadata():
    torch.manual_seed(41)
    page_size = 256
    tokens, heads, topk = 1, 8, 64
    raw = _make_flashmla_raw_cache(page_size, blocks=1, cache_tokens=256, seed=43)
    q = torch.randn((tokens, heads, 512), device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(topk, device="cuda", dtype=torch.int32).reshape(1, topk).repeat(tokens, 1)
    topk_lengths = torch.full((tokens,), topk, device="cuda", dtype=torch.int32)
    attn_sink = torch.zeros((heads,), device="cuda", dtype=torch.float32)

    out_first = torch.empty_like(q)
    _, sched_meta, num_splits = infinicore.deepseek_v4_flashmla_sparse_attention_with_metadata_(
        _as_core(q),
        _as_core(raw),
        _as_core(indices),
        _as_core(topk_lengths),
        _as_core(attn_sink),
        _as_core(out_first),
        None,
        None,
        512**-0.5,
        page_size,
        512,
    )
    infinicore.sync_stream()
    assert sched_meta.shape[1] == 8
    assert num_splits.numel() == 2

    out_second = torch.empty_like(q)
    _, sched_meta_again, num_splits_again = infinicore.deepseek_v4_flashmla_sparse_attention_with_metadata_(
        _as_core(q),
        _as_core(raw),
        _as_core(indices),
        _as_core(topk_lengths),
        _as_core(attn_sink),
        _as_core(out_second),
        sched_meta,
        num_splits,
        512**-0.5,
        page_size,
        512,
    )
    infinicore.sync_stream()
    assert torch.equal(out_first, out_second)
    assert sched_meta_again.data_ptr() == sched_meta.data_ptr()
    assert num_splits_again.data_ptr() == num_splits.data_ptr()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.parse_args()
    test_metadata_object()
    test_dense_fp8_metadata()
    test_sparse_prefill_compute()
    test_sparse_decode_attention_compute()
    test_sparse_decode_local_heads_match_full_heads()
    test_sparse_decode_attention_cached_metadata()
    print("DeepseekV4FlashMLACompute: passed")


if __name__ == "__main__":
    main()
