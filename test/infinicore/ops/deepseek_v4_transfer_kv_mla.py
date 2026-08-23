import argparse

import infinicore
import torch
from sgl_kernel import kvcacheio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.parse_args()

    src_indices = torch.tensor([0, 2], device="cuda", dtype=torch.int64)
    dst_indices = torch.tensor([1, 3], device="cuda", dtype=torch.int64)
    item_size = 16

    src = torch.arange(4 * item_size, device="cuda", dtype=torch.uint8).reshape(4, item_size)
    ref_dst = torch.zeros(4, item_size, device="cuda", dtype=torch.uint8)
    kvcacheio.transfer_kv_per_layer_mla(src, ref_dst, src_indices, dst_indices, item_size, 1, 1)
    torch.cuda.synchronize()

    out_dst = torch.zeros_like(ref_dst)
    infinicore.deepseek_v4_transfer_kv_per_layer_mla_(
        infinicore.from_torch(src),
        infinicore.from_torch(out_dst),
        infinicore.from_torch(src_indices),
        infinicore.from_torch(dst_indices),
        item_size,
        1,
        1,
    )
    infinicore.sync_stream()
    assert torch.equal(out_dst, ref_dst)

    src_pf = torch.arange(2 * 4 * item_size, device="cuda", dtype=torch.uint8).reshape(2, 4, item_size)
    ref_dst_pf_lf = torch.zeros(4, item_size, device="cuda", dtype=torch.uint8)
    kvcacheio.transfer_kv_per_layer_mla_pf_lf(
        src_pf, ref_dst_pf_lf, src_indices, dst_indices, 1, item_size, 4, 1, 1
    )
    torch.cuda.synchronize()

    out_dst_pf_lf = torch.zeros_like(ref_dst_pf_lf)
    infinicore.deepseek_v4_transfer_kv_per_layer_mla_pf_lf_(
        infinicore.from_torch(src_pf),
        infinicore.from_torch(out_dst_pf_lf),
        infinicore.from_torch(src_indices),
        infinicore.from_torch(dst_indices),
        1,
        item_size,
        4,
        1,
        1,
    )
    infinicore.sync_stream()
    assert torch.equal(out_dst_pf_lf, ref_dst_pf_lf)
    print("DeepseekV4TransferKVMLA: passed")


if __name__ == "__main__":
    main()
