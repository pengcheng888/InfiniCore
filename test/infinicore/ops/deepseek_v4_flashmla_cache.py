import argparse

import infinicore
import sgl_kernel.flash_mla
import torch
import vllm._C  # noqa: F401


def _as_core(tensor):
    return infinicore.from_torch(tensor)


def _test_fused_store(device):
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
    infinicore.deepseek_v4_fused_store_flashmla_cache_(
        _as_core(kv_c),
        _as_core(k_pe),
        _as_core(out_cache),
        _as_core(slot_mapping),
        "auto",
        _as_core(scale),
    )
    infinicore.sync_stream()
    assert torch.equal(out_cache, ref_cache)


def _test_indexer(device):
    req_to_token = torch.arange(4 * 128, dtype=torch.int32, device=device).reshape(4, 128)
    req_pool_indices = torch.tensor([0, 1, 3], dtype=torch.int32, device=device)
    page_kernel_lens = torch.tensor([1, 65, 128], dtype=torch.int32, device=device)
    max_pages = 2
    page_size = 64

    ref = torch.full((req_pool_indices.numel(), max_pages), -1, dtype=torch.int32, device=device)
    sgl_kernel.flash_mla.dcu_create_flashmla_kv_indices(
        req_to_token,
        req_pool_indices,
        page_kernel_lens,
        None,
        ref,
        req_to_token.stride(0),
        max_pages,
        page_size,
    )
    torch.cuda.synchronize()

    out = torch.full_like(ref, -1)
    infinicore.deepseek_v4_flashmla_cache_indexer_(
        _as_core(req_to_token),
        _as_core(req_pool_indices),
        _as_core(page_kernel_lens),
        None,
        _as_core(out),
        req_to_token.stride(0),
        max_pages,
        page_size,
    )
    infinicore.sync_stream()
    assert torch.equal(out, ref)


def _hadamard_ref(x):
    work = x.reshape(-1, x.shape[-1]).float().contiguous()
    dim = work.shape[-1]
    span = 1
    while span < dim:
        view = work.reshape(work.shape[0], dim // (2 * span), 2, span)
        even = view[:, :, 0, :].clone()
        odd = view[:, :, 1, :].clone()
        view[:, :, 0, :].copy_(even + odd)
        view[:, :, 1, :].copy_(even - odd)
        span *= 2
    work.mul_(dim ** -0.5)
    return work.reshape_as(x)


def _test_indexer_rotate_and_store(device):
    torch.manual_seed(1)
    page_size = 64
    blocks = 2
    x = torch.randn(3, 128, dtype=torch.float32, device=device)
    rotated = x.clone()
    infinicore.deepseek_v4_indexer_rotate_(_as_core(rotated))
    infinicore.sync_stream()
    assert torch.allclose(rotated, _hadamard_ref(x), atol=1e-5, rtol=1e-5)

    indices = torch.tensor([0, 63, 64], dtype=torch.int32, device=device)
    out_cache = torch.zeros(blocks, 132 * page_size, dtype=torch.uint8, device=device)
    infinicore.deepseek_v4_store_indexer_raw_cache_(
        _as_core(rotated),
        _as_core(out_cache),
        _as_core(indices),
        page_size,
    )
    infinicore.sync_stream()

    scale = torch.clamp(rotated.abs().amax(dim=-1, keepdim=True), min=1.0e-4) / 448.0
    quant = torch.clamp(rotated / scale, -448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    scale_bytes = scale.reshape(-1).contiguous().view(torch.uint8).reshape(-1, 4)
    ref_cache = torch.zeros_like(out_cache)
    flat = ref_cache.reshape(-1)
    for row, idx in enumerate(indices.cpu().tolist()):
        page = idx // page_size
        offset = idx % page_size
        base = page * 132 * page_size + offset * 128
        scale_base = page * 132 * page_size + 128 * page_size + offset * 4
        flat[base : base + 128] = quant[row]
        flat[scale_base : scale_base + 4] = scale_bytes[row]
    assert torch.equal(out_cache, ref_cache)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.parse_args()
    device = "cuda"

    _test_fused_store(device)
    _test_indexer(device)
    _test_indexer_rotate_and_store(device)
    print("DeepseekV4FlashMLACache: passed")


if __name__ == "__main__":
    main()
