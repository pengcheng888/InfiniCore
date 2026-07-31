import torch

import flash_mla.cuda as flash_mla_cuda
import infinicore
from infinicore.lib import _infinicore


CUDA_ENTRYPOINTS = [
    "dense_decode_fwd",
    "dense_decode_fwd_kvfp8",
    "dense_decode_fwd_qkvfp8",
    "fwd_kvcache_mla_fp8",
    "fwd_kvcache_mla_fp8_with_cat",
    "fwd_kvcache_mla_nope_pe",
    "fwd_kvcache_quantization_mla",
    "fwd_kvcache_quantization_q_nope_pe_mla",
    "get_mla_decoding_metadata_dense_fp8",
    "sparse_decode_fwd",
    "sparse_prefill_fwd",
]


def _deepseek_name(name):
    return f"deepseek_v4_{name}"


def test_exports():
    for name in CUDA_ENTRYPOINTS:
        deepseek_name = _deepseek_name(name)
        assert callable(getattr(infinicore, deepseek_name))
        assert callable(getattr(_infinicore, deepseek_name))


def test_no_arg_forwarding_reaches_flashmla():
    for name in CUDA_ENTRYPOINTS:
        fn = getattr(infinicore, _deepseek_name(name))
        try:
            fn()
        except TypeError as exc:
            msg = str(exc)
            assert name in msg or "incompatible function arguments" in msg
        else:
            raise AssertionError(f"{name} unexpectedly accepted no arguments")


def test_get_metadata_dense_fp8_matches_flashmla():
    cache_seqlens = torch.tensor([64, 96], device="cuda", dtype=torch.int32)

    got = infinicore.deepseek_v4_get_mla_decoding_metadata_dense_fp8(cache_seqlens, 16, 1)
    ref = flash_mla_cuda.get_mla_decoding_metadata_dense_fp8(cache_seqlens, 16, 1)

    assert len(got) == len(ref)
    for got_item, ref_item in zip(got, ref):
        assert torch.equal(got_item, ref_item)


def test_sparse_prefill_matches_flashmla():
    torch.manual_seed(7)
    seq_q = 2
    seq_kv = 128
    heads_q = 2
    heads_kv = 1
    topk = 128
    dim = 576

    q = torch.randn((seq_q, heads_q, dim), device="cuda", dtype=torch.bfloat16)
    kv = torch.randn((seq_kv, heads_kv, dim), device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(topk, device="cuda", dtype=torch.int32).reshape(1, 1, topk).repeat(seq_q, heads_kv, 1)
    sm_scale = dim ** -0.5

    got = infinicore.deepseek_v4_sparse_prefill_fwd(q, kv, indices, sm_scale, 512, None, None)
    ref = flash_mla_cuda.sparse_prefill_fwd(q, kv, indices, sm_scale, 512, None, None)
    torch.cuda.synchronize()

    assert len(got) == len(ref)
    for got_item, ref_item in zip(got, ref):
        assert torch.equal(got_item, ref_item)


if __name__ == "__main__":
    test_exports()
    test_no_arg_forwarding_reaches_flashmla()
    test_get_metadata_dense_fp8_matches_flashmla()
    test_sparse_prefill_matches_flashmla()
    print("deepseek_v4 flash_mla.cuda entrypoint tests passed")
