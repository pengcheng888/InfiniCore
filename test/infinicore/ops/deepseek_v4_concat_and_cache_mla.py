import argparse

import infinicore
import torch
import vllm._C  # noqa: F401


def _device(args):
    return "cuda" if args.hygon or args.nvidia else "cuda"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    args = parser.parse_args()
    device = _device(args)

    torch.manual_seed(0)
    batch, kv_c_dim, rot_dim, blocks, block_size = 3, 8, 4, 2, 4
    kv_c = torch.randn(batch, kv_c_dim, dtype=torch.bfloat16, device=device)
    k_pe = torch.randn(batch, rot_dim, dtype=torch.bfloat16, device=device)
    slot_mapping = torch.tensor([0, 3, 5], dtype=torch.int64, device=device)
    scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    ref_cache = torch.zeros(blocks, block_size, kv_c_dim + rot_dim, dtype=torch.bfloat16, device=device)
    torch.ops._C_cache_ops.concat_and_cache_mla(kv_c, k_pe, ref_cache, slot_mapping, "auto", scale)
    torch.cuda.synchronize()

    out_cache = torch.zeros_like(ref_cache)
    infinicore.deepseek_v4_concat_and_cache_mla_(
        infinicore.from_torch(kv_c),
        infinicore.from_torch(k_pe),
        infinicore.from_torch(out_cache),
        infinicore.from_torch(slot_mapping),
        "auto",
        infinicore.from_torch(scale),
    )
    infinicore.sync_stream()

    assert torch.equal(out_cache, ref_cache)
    print("DeepseekV4ConcatAndCacheMLA: passed")


if __name__ == "__main__":
    main()
