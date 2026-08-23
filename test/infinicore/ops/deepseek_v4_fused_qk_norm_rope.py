import argparse

import infinicore
import torch
import vllm._C  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.parse_args()

    torch.manual_seed(4)
    tokens, num_heads_q, num_heads_k, head_dim = 5, 4, 2, 64
    num_heads_v = num_heads_k
    qkv = torch.randn(tokens, (num_heads_q + num_heads_k + num_heads_v) * head_dim, device="cuda", dtype=torch.bfloat16)
    q_weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
    cos_sin_cache = torch.randn(128, head_dim, device="cuda", dtype=torch.bfloat16)
    position_ids = torch.arange(tokens, device="cuda", dtype=torch.int64)

    ref_qkv = qkv.clone()
    torch.ops._C.fused_qk_norm_rope(
        ref_qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        cos_sin_cache,
        False,
        position_ids,
    )
    torch.cuda.synchronize()

    out_qkv = qkv.clone()
    infinicore.deepseek_v4_fused_qk_norm_rope_(
        infinicore.from_torch(out_qkv),
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        1e-6,
        infinicore.from_torch(q_weight),
        infinicore.from_torch(k_weight),
        infinicore.from_torch(cos_sin_cache),
        False,
        infinicore.from_torch(position_ids),
    )
    infinicore.sync_stream()

    assert torch.allclose(out_qkv, ref_qkv, atol=0, rtol=0)
    print("DeepseekV4FusedQKNormRoPE: passed")


if __name__ == "__main__":
    main()
