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
    allocate_lens = torch.tensor([0, 3], device=device, dtype=torch.int32)
    new_allocate_lens = torch.tensor([2, 1], device=device, dtype=torch.int32)
    cache_locs = torch.tensor([100, 101, 102], device=device, dtype=torch.int64)

    ref_req_to_token = torch.full((4, 8), -1, device=device, dtype=torch.int32)
    ref_cache_locs = cache_locs.clone()
    kvcacheio.dcu_assign_req_to_token_pool(
        req_pool_indices,
        ref_req_to_token,
        allocate_lens,
        new_allocate_lens,
        ref_cache_locs,
        cache_locs.numel(),
        req_pool_indices.numel(),
    )
    torch.cuda.synchronize()

    out_req_to_token = torch.full_like(ref_req_to_token, -1)
    out_cache_locs = cache_locs.clone()
    infinicore.deepseek_v4_assign_req_to_token_pool_(
        infinicore.from_torch(req_pool_indices),
        infinicore.from_torch(out_req_to_token),
        infinicore.from_torch(allocate_lens),
        infinicore.from_torch(new_allocate_lens),
        infinicore.from_torch(out_cache_locs),
        cache_locs.numel(),
        req_pool_indices.numel(),
    )
    infinicore.sync_stream()

    assert torch.equal(out_req_to_token, ref_req_to_token)
    assert torch.equal(out_cache_locs, ref_cache_locs)
    print("DeepseekV4AssignReqToTokenPool: passed")


if __name__ == "__main__":
    main()
