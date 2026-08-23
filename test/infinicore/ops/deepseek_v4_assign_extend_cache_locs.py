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

    req_pool_indices = torch.tensor([0, 1], device=device, dtype=torch.int64)
    req_to_token = torch.arange(4 * 8, device=device, dtype=torch.int32).reshape(4, 8)
    start_offset = torch.tensor([1, 2], device=device, dtype=torch.int64)
    end_offset = torch.tensor([4, 5], device=device, dtype=torch.int64)

    ref_out = torch.full((6,), -1, device=device, dtype=torch.int64)
    kvcacheio.dcu_assign_extend_cache_locs(
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        ref_out,
        req_to_token.shape[1],
        req_pool_indices.numel(),
    )
    torch.cuda.synchronize()

    out = torch.full_like(ref_out, -1)
    infinicore.deepseek_v4_assign_extend_cache_locs_(
        infinicore.from_torch(req_pool_indices),
        infinicore.from_torch(req_to_token),
        infinicore.from_torch(start_offset),
        infinicore.from_torch(end_offset),
        infinicore.from_torch(out),
        req_to_token.shape[1],
        req_pool_indices.numel(),
    )
    infinicore.sync_stream()

    assert torch.equal(out, ref_out)
    print("DeepseekV4AssignExtendCacheLocs: passed")


if __name__ == "__main__":
    main()
