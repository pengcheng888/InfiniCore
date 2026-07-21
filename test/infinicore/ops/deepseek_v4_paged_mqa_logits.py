import argparse
import ctypes

import deepgemm
import deepgemm.op
import infinicore
import torch


def _as_core(tensor):
    return infinicore.from_torch(tensor)


def _load_deepgemm_global():
    ctypes.CDLL(deepgemm.op.__file__, mode=ctypes.RTLD_GLOBAL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.parse_args()
    _load_deepgemm_global()

    torch.manual_seed(0)
    device = "cuda"
    batch, next_n, heads, head_dim = 2, 1, 32, 128
    block_kv, num_blocks, max_context_len, num_sms = 64, 4, 128, 80

    q = torch.randn(batch, next_n, heads, head_dim, device=device, dtype=torch.bfloat16) * 0.01
    fused_kv_cache = torch.randn(num_blocks, block_kv, 1, head_dim, device=device, dtype=torch.bfloat16) * 0.01
    weights = torch.ones((batch, heads), device=device, dtype=torch.float32)
    context_lens = torch.tensor([64, 128], device=device, dtype=torch.int32)
    block_table = torch.tensor([[0, 1], [2, 3]], device=device, dtype=torch.int32)

    ref_meta = deepgemm.get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms)
    torch.cuda.synchronize()
    out_meta = torch.empty_like(ref_meta)
    infinicore.deepseek_v4_paged_mqa_logits_metadata_(
        _as_core(context_lens),
        _as_core(out_meta),
        block_kv,
        num_sms,
    )
    infinicore.sync_stream()
    assert torch.equal(out_meta, ref_meta)

    ref = deepgemm.paged_mqa_logits(
        q,
        fused_kv_cache,
        weights,
        context_lens,
        block_table,
        ref_meta,
        max_context_len,
        True,
    )
    torch.cuda.synchronize()
    out = torch.empty_like(ref)
    infinicore.deepseek_v4_paged_mqa_logits_(
        _as_core(q),
        _as_core(fused_kv_cache),
        _as_core(weights),
        _as_core(context_lens),
        _as_core(block_table),
        _as_core(out_meta),
        _as_core(out),
        max_context_len,
        True,
    )
    infinicore.sync_stream()
    assert torch.equal(out, ref)

    print("DeepseekV4PagedMQALogits: passed")


if __name__ == "__main__":
    main()
