import argparse

import infinicore
import torch
from sgl_kernel import kvcacheio


def _as_core(tensor):
    return infinicore.from_torch(tensor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.parse_args()

    src_indices = torch.tensor([0, 2], device="cuda", dtype=torch.int64)
    dst_indices = torch.tensor([1, 3], device="cuda", dtype=torch.int64)
    item_size = 16

    src_k = torch.arange(4 * item_size, device="cuda", dtype=torch.uint8).reshape(4, item_size)
    src_v = (torch.arange(4 * item_size, device="cuda", dtype=torch.uint8) + 80).reshape(4, item_size)
    ref_dst_k = torch.zeros_like(src_k)
    ref_dst_v = torch.zeros_like(src_v)
    kvcacheio.transfer_kv_per_layer(src_k, ref_dst_k, src_v, ref_dst_v, src_indices, dst_indices, item_size, 1, 1)
    torch.cuda.synchronize()

    out_dst_k = torch.zeros_like(ref_dst_k)
    out_dst_v = torch.zeros_like(ref_dst_v)
    infinicore.deepseek_v4_transfer_kv_per_layer_(
        _as_core(src_k),
        _as_core(out_dst_k),
        _as_core(src_v),
        _as_core(out_dst_v),
        _as_core(src_indices),
        _as_core(dst_indices),
        item_size,
        1,
        1,
    )
    infinicore.sync_stream()
    assert torch.equal(out_dst_k, ref_dst_k)
    assert torch.equal(out_dst_v, ref_dst_v)

    src_k_pf = torch.arange(2 * 4 * item_size, device="cuda", dtype=torch.uint8).reshape(2, 4, item_size)
    src_v_pf = (torch.arange(2 * 4 * item_size, device="cuda", dtype=torch.uint8) + 70).reshape(2, 4, item_size)
    ref_dst_k_pf_lf = torch.zeros(4, item_size, device="cuda", dtype=torch.uint8)
    ref_dst_v_pf_lf = torch.zeros_like(ref_dst_k_pf_lf)
    kvcacheio.transfer_kv_per_layer_pf_lf(
        src_k_pf,
        ref_dst_k_pf_lf,
        src_v_pf,
        ref_dst_v_pf_lf,
        src_indices,
        dst_indices,
        1,
        item_size,
        4,
        1,
        1,
    )
    torch.cuda.synchronize()

    out_dst_k_pf_lf = torch.zeros_like(ref_dst_k_pf_lf)
    out_dst_v_pf_lf = torch.zeros_like(ref_dst_v_pf_lf)
    infinicore.deepseek_v4_transfer_kv_per_layer_pf_lf_(
        _as_core(src_k_pf),
        _as_core(out_dst_k_pf_lf),
        _as_core(src_v_pf),
        _as_core(out_dst_v_pf_lf),
        _as_core(src_indices),
        _as_core(dst_indices),
        1,
        item_size,
        4,
        1,
        1,
    )
    infinicore.sync_stream()
    assert torch.equal(out_dst_k_pf_lf, ref_dst_k_pf_lf)
    assert torch.equal(out_dst_v_pf_lf, ref_dst_v_pf_lf)

    print("DeepseekV4TransferKV: passed")


if __name__ == "__main__":
    main()
