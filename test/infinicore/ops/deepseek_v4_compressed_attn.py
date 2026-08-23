import os

os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")

import argparse

import infinicore
import torch
from sglang.jit_kernel import deepseek_v4


def _as_core(tensor):
    return infinicore.from_torch(tensor)


def _same_plan(a, b):
    assert type(a) is type(b)
    assert a.compress_ratio == b.compress_ratio
    for lhs, rhs in zip(a[1:], b[1:]):
        assert torch.equal(lhs, rhs)


def test_metadata_decode():
    seq_lens = torch.tensor([4, 8], device="cuda", dtype=torch.int32)
    ref = deepseek_v4.compress_plan(4, 2, seq_lens, None, seq_lens.device)
    got = infinicore.deepseek_v4_compressed_attn_metadata_(4, 2, _as_core(seq_lens))
    _same_plan(got, ref)


def test_metadata_prefill():
    seq_lens = torch.tensor([8, 12], device="cuda", dtype=torch.int32)
    extend_lens = torch.tensor([4, 4], device="cuda", dtype=torch.int32)
    ref = deepseek_v4.compress_plan(4, 8, seq_lens, extend_lens, seq_lens.device)
    got = infinicore.deepseek_v4_compressed_attn_metadata_(
        4, 8, _as_core(seq_lens), _as_core(extend_lens)
    )
    _same_plan(got, ref)


def test_decode_compute():
    torch.manual_seed(3)
    batch, head_dim, ratio = 2, 128, 4
    kv_score_buffer = torch.randn((2, 8, head_dim * ratio), device="cuda", dtype=torch.float32)
    kv_score_input = torch.randn((batch, head_dim * ratio), device="cuda", dtype=torch.float32)
    ape = torch.randn((8, head_dim), device="cuda", dtype=torch.float32)
    indices = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([4, 8], device="cuda", dtype=torch.int32)
    ref_plan = deepseek_v4.compress_plan(ratio, batch, seq_lens, None, seq_lens.device)
    ref = deepseek_v4.compress_forward(
        kv_score_buffer,
        kv_score_input,
        ape,
        indices,
        plan=ref_plan,
        head_dim=head_dim,
        compress_ratio=ratio,
    )
    torch.cuda.synchronize()

    out = torch.empty_like(ref)
    got_plan = infinicore.deepseek_v4_compressed_attn_metadata_(ratio, batch, _as_core(seq_lens))
    infinicore.deepseek_v4_compressed_attn_decode_(
        _as_core(kv_score_buffer),
        _as_core(kv_score_input),
        _as_core(ape),
        _as_core(indices),
        _as_core(out),
        got_plan,
        head_dim=head_dim,
        compress_ratio=ratio,
        seq_lens=_as_core(seq_lens),
    )
    infinicore.sync_stream()
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.parse_args()
    test_metadata_decode()
    test_metadata_prefill()
    test_decode_compute()
    print("DeepseekV4CompressedAttn: passed")


if __name__ == "__main__":
    main()
