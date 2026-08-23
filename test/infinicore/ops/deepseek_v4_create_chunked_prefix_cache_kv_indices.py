import argparse

import infinicore
import torch
from sgl_kernel import kvcacheio


def _device(args):
    return "cuda" if args.hygon or args.nvidia else "cuda"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    args = parser.parse_args()
    device = _device(args)

    req_to_token = torch.arange(4 * 16, device=device, dtype=torch.int32).reshape(4, 16)
    req_pool_indices = torch.tensor([0, 2], device=device, dtype=torch.int64)
    chunk_starts = torch.tensor([1, 3], device=device, dtype=torch.int32)
    chunk_seq_lens = torch.tensor([4, 5], device=device, dtype=torch.int32)
    chunk_cu_seq_lens = torch.tensor([0, 4, 9], device=device, dtype=torch.int32)

    ref_out = torch.full((2, 8), -1, device=device, dtype=torch.int32)
    kvcacheio.dcu_create_chunked_prefix_cache_kv_indices(
        req_to_token,
        req_pool_indices,
        chunk_starts,
        chunk_seq_lens,
        chunk_cu_seq_lens,
        ref_out,
        ref_out.shape[1],
        req_pool_indices.numel(),
    )
    torch.cuda.synchronize()

    out = torch.full_like(ref_out, -1)
    infinicore.deepseek_v4_create_chunked_prefix_cache_kv_indices_(
        infinicore.from_torch(req_to_token),
        infinicore.from_torch(req_pool_indices),
        infinicore.from_torch(chunk_starts),
        infinicore.from_torch(chunk_seq_lens),
        infinicore.from_torch(chunk_cu_seq_lens),
        infinicore.from_torch(out),
        out.shape[1],
        req_pool_indices.numel(),
    )
    infinicore.sync_stream()

    assert torch.equal(out, ref_out)
    print("DeepseekV4CreateChunkedPrefixCacheKVIndices: passed")


if __name__ == "__main__":
    main()
