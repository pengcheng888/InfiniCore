import argparse
import time

import infinicore
import torch
from infinicore.lib import _infinicore


FP8_MAX = 448.0
TOPK = 512


def _as_core(tensor):
    return infinicore.from_torch(tensor)._underlying


def _sync():
    infinicore.sync_stream()
    torch.cuda.synchronize()


def _bench(name, ref_fn, native_fn, repeats, warmup):
    for _ in range(warmup):
        ref_fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(repeats):
        ref_fn()
    _sync()
    ref_ms = (time.perf_counter() - t0) * 1000.0 / repeats

    for _ in range(warmup):
        native_fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(repeats):
        native_fn()
    _sync()
    native_ms = (time.perf_counter() - t0) * 1000.0 / repeats

    speedup = ref_ms / native_ms if native_ms > 0 else float("inf")
    print(f"{name:34s} ref_at={ref_ms:8.4f} ms  native={native_ms:8.4f} ms  speedup={speedup:6.2f}x")


def _flash_page_bytes(page_size):
    value_bytes = 576
    scale_bytes = 8
    raw = (value_bytes + scale_bytes) * page_size
    return ((raw + value_bytes - 1) // value_bytes) * value_bytes


def _indexer_page_bytes(page_size):
    return (128 + 4) * page_size


def _hadamard_ref_(x, apply_scale=True):
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
    if apply_scale:
        work.mul_(dim ** -0.5)
    x.copy_(work.reshape_as(x).to(x.dtype))
    return x


def _topk_transform_ref(scores, seq_lens, page_table, page_size):
    batch, max_seq_len = scores.shape
    seq = seq_lens.to(torch.int32)
    sequential = torch.arange(TOPK, device=scores.device, dtype=torch.int32).unsqueeze(0).expand(batch, TOPK)
    negative = torch.full((batch, TOPK), -1, device=scores.device, dtype=torch.int32)
    sequential_valid = sequential < seq.unsqueeze(1)

    if max_seq_len <= TOPK:
        raw_indices = torch.where(sequential_valid, sequential, negative)
        valid_topk = sequential_valid
    else:
        positions = torch.arange(max_seq_len, device=scores.device, dtype=torch.int64).unsqueeze(0)
        valid_mask = positions < seq.to(torch.int64).unsqueeze(1)
        masked_scores = scores.masked_fill(~valid_mask, -float("inf"))
        raw_indices = torch.topk(masked_scores, TOPK, dim=1, largest=True, sorted=False).indices.to(torch.int32)
        gathered_scores = scores.gather(1, raw_indices.to(torch.int64))
        valid_topk = gathered_scores.ne(-float("inf"))
        needs_sequential = (seq <= TOPK).unsqueeze(1)
        raw_indices = torch.where(needs_sequential, torch.where(sequential_valid, sequential, negative), raw_indices)
        valid_topk = torch.where(needs_sequential, sequential_valid, valid_topk)

    raw_long = raw_indices.to(torch.int64)
    page_idx = torch.div(raw_long, page_size, rounding_mode="floor")
    offset = torch.remainder(raw_long, page_size)
    physical_pages = page_table.to(torch.int64).gather(1, torch.clamp_min(page_idx, 0))
    page_indices = (physical_pages * page_size + offset).to(torch.int32)
    return torch.where(valid_topk, page_indices, negative)


def _act_quant_ref(q, weights, weight_scale):
    q_contig = q.contiguous()
    q_float = q_contig.reshape(-1, 128).float()
    scale = torch.clamp(q_float.abs().amax(dim=-1, keepdim=True), min=1.0e-4) / FP8_MAX
    q_fp8 = torch.clamp(q_float / scale, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).reshape_as(q_contig)
    q_scale = scale.reshape(q_contig.shape[0], q_contig.shape[1], 1)
    fused = weights.contiguous().float() * weight_scale * q_scale.squeeze(2).float()
    return q_fp8, q_scale, fused


def _check_rotate(device):
    torch.manual_seed(11)
    x = (torch.randn(17, 128, device=device, dtype=torch.bfloat16) * 0.2).contiguous()
    ref = x.clone()
    out = x.clone()
    _infinicore.deepseek_v4_indexer_rotate_naive_(_as_core(ref), True)
    _infinicore.deepseek_v4_indexer_rotate_128_kernel_(_as_core(out), True)
    _sync()
    assert torch.equal(out, ref), f"rotate mismatch max={((out.float() - ref.float()).abs().max()).item()}"


def _check_indexer_store(device):
    torch.manual_seed(12)
    page_size = 64
    tokens = 9
    blocks = 2
    x = (torch.randn(tokens, 128, device=device, dtype=torch.bfloat16) * 0.3).contiguous()
    indices = torch.tensor([0, 1, -1, 7, 63, 64, 65, 79, -1], device=device, dtype=torch.int32)
    ref = torch.zeros(blocks, _indexer_page_bytes(page_size), device=device, dtype=torch.uint8)
    out = torch.zeros_like(ref)
    _infinicore.deepseek_v4_store_indexer_raw_cache_naive_(_as_core(x), _as_core(ref), _as_core(indices), page_size)
    _infinicore.deepseek_v4_store_indexer_raw_cache_kernel_(_as_core(x), _as_core(out), _as_core(indices), page_size)
    _sync()
    assert torch.equal(out, ref), f"indexer raw cache mismatch bytes={(out != ref).sum().item()}"


def _check_flash_store(device):
    torch.manual_seed(13)
    page_size = 64
    tokens = 7
    blocks = 2
    x = (torch.randn(tokens, 512, device=device, dtype=torch.bfloat16) * 0.25).contiguous()
    indices = torch.tensor([0, -1, 5, 63, 64, 65, 79], device=device, dtype=torch.int64)
    ref = torch.zeros(blocks, _flash_page_bytes(page_size), device=device, dtype=torch.uint8)
    out = torch.zeros_like(ref)
    _infinicore.deepseek_v4_store_flashmla_raw_cache_naive_(_as_core(x), _as_core(ref), _as_core(indices), page_size)
    _infinicore.deepseek_v4_store_flashmla_raw_cache_kernel_(_as_core(x), _as_core(out), _as_core(indices), page_size)
    _sync()
    assert torch.equal(out, ref), f"flash raw cache mismatch bytes={(out != ref).sum().item()}"


def _check_act_quant(device):
    torch.manual_seed(14)
    batch, heads = 19, 32
    q = (torch.randn(batch, heads, 128, device=device, dtype=torch.bfloat16) * 0.2).contiguous()
    weights = torch.randn(batch, heads, device=device, dtype=torch.bfloat16).contiguous()
    weight_scale = 0.375
    ref_fp8, ref_scale, ref_fused = _act_quant_ref(q, weights, weight_scale)
    out_fp8 = torch.empty_like(ref_fp8)
    out_scale = torch.empty_like(ref_scale)
    out_fused = torch.empty_like(ref_fused)
    _infinicore.deepseek_v4_c4_act_quant_fused_scale_kernel_(
        _as_core(q),
        _as_core(weights),
        _as_core(out_fp8),
        _as_core(out_scale),
        _as_core(out_fused),
        weight_scale,
    )
    _sync()
    assert torch.equal(out_fp8.view(torch.uint8), ref_fp8.view(torch.uint8)), "act quant fp8 bytes mismatch"
    assert torch.allclose(out_scale, ref_scale, atol=0.0, rtol=0.0), "act quant scale mismatch"
    assert torch.allclose(out_fused, ref_fused, atol=1e-7, rtol=1e-6), "act quant fused weights mismatch"


def _check_topk(device):
    torch.manual_seed(15)
    page_size = 64
    batch, max_seq_len = 6, 384
    scores = torch.randn(batch, max_seq_len, device=device, dtype=torch.float32).contiguous()
    seq_lens = torch.tensor([0, 1, 17, 64, 255, 384], device=device, dtype=torch.int32)
    pages = (max_seq_len + page_size - 1) // page_size
    page_table = (torch.arange(batch * pages, device=device, dtype=torch.int32).reshape(batch, pages) + 11).contiguous()
    ref = _topk_transform_ref(scores, seq_lens, page_table, page_size)
    out = torch.empty(batch, TOPK, device=device, dtype=torch.int32)
    _infinicore.deepseek_v4_topk_transform_512_kernel_(_as_core(scores), _as_core(seq_lens), _as_core(page_table), _as_core(out), page_size)
    _sync()
    assert torch.equal(out, ref), "topk transform <=512 mismatch"

    max_seq_len = 640
    scores = torch.randn(3, max_seq_len, device=device, dtype=torch.float32).contiguous()
    seq_lens = torch.tensor([12, 511, 640], device=device, dtype=torch.int64)
    pages = (max_seq_len + page_size - 1) // page_size
    page_table = torch.arange(3 * pages, device=device, dtype=torch.int64).reshape(3, pages).contiguous()
    ref = _topk_transform_ref(scores, seq_lens, page_table, page_size)
    out = torch.empty(3, TOPK, device=device, dtype=torch.int32)
    _infinicore.deepseek_v4_topk_transform_512_kernel_(_as_core(scores), _as_core(seq_lens), _as_core(page_table), _as_core(out), page_size)
    _sync()
    assert torch.equal(out[:2], ref[:2]), "topk transform sequential large mismatch"
    assert torch.equal(torch.sort(out[2]).values, torch.sort(ref[2]).values), "topk transform >512 set mismatch"


def _check_c4_indexer_split_chain(device):
    _check_c4_indexer_split_chain_case(device, pages=4)
    _check_c4_indexer_split_chain_case(device, pages=16)


def _check_c4_indexer_split_chain_case(device, pages):
    torch.manual_seed(16)
    page_size = 64
    batch, heads = 5, 32
    max_c4_seq_len = pages * page_size
    blocks = batch * pages

    q = (torch.randn(batch, heads, 128, device=device, dtype=torch.bfloat16) * 0.2).contiguous()
    weights = torch.randn(batch, heads, device=device, dtype=torch.bfloat16).contiguous()
    cache_raw = torch.zeros(blocks, page_size * (128 + 4), device=device, dtype=torch.uint8).contiguous()
    cache_values = (torch.randn(blocks * page_size, 128, device=device, dtype=torch.bfloat16) * 0.2).contiguous()
    cache_indices = torch.arange(blocks * page_size, device=device, dtype=torch.int32)
    _infinicore.deepseek_v4_store_indexer_raw_cache_kernel_(
        _as_core(cache_values),
        _as_core(cache_raw),
        _as_core(cache_indices),
        page_size,
    )
    seq_lens = torch.tensor([0, 1, 73, 128, max_c4_seq_len], device=device, dtype=torch.int32)
    page_table = torch.arange(blocks, device=device, dtype=torch.int32).reshape(batch, pages).contiguous()
    weight_scale = 0.375

    ref_logits = torch.empty(batch, max_c4_seq_len, device=device, dtype=torch.float32)
    ref_indices = torch.empty(batch, TOPK, device=device, dtype=torch.int32)
    _infinicore.deepseek_v4_c4_sparse_attn_indexer_(
        _as_core(q),
        _as_core(weights),
        _as_core(cache_raw),
        _as_core(seq_lens),
        _as_core(page_table),
        _as_core(ref_logits),
        _as_core(ref_indices),
        max_c4_seq_len,
        page_size,
        weight_scale,
        False,
    )

    q_fp8 = torch.empty(batch, heads, 128, device=device, dtype=torch.float8_e4m3fn)
    q_scale = torch.empty(batch, heads, 1, device=device, dtype=torch.float32)
    fused_weights = torch.empty(batch, heads, device=device, dtype=torch.float32)
    logits = torch.empty_like(ref_logits)
    indices = torch.empty_like(ref_indices)
    fused_indices = torch.empty_like(ref_indices)
    _infinicore.deepseek_v4_c4_act_quant_fused_scale_kernel_(
        _as_core(q),
        _as_core(weights),
        _as_core(q_fp8),
        _as_core(q_scale),
        _as_core(fused_weights),
        weight_scale,
    )
    _infinicore.deepseek_v4_c4_paged_mqa_logits_(
        _as_core(q_fp8),
        _as_core(fused_weights),
        _as_core(cache_raw),
        _as_core(seq_lens),
        _as_core(page_table),
        _as_core(logits),
        max_c4_seq_len,
        page_size,
        False,
    )
    _infinicore.deepseek_v4_topk_transform_512_kernel_(
        _as_core(logits),
        _as_core(seq_lens),
        _as_core(page_table),
        _as_core(indices),
        page_size,
    )
    _infinicore.deepseek_v4_c4_paged_mqa_with_topk_transform_512_(
        _as_core(q_fp8),
        _as_core(fused_weights),
        _as_core(cache_raw),
        _as_core(seq_lens),
        _as_core(page_table),
        _as_core(fused_indices),
        max_c4_seq_len,
        page_size,
        False,
    )
    _sync()
    for row, seq_len in enumerate(seq_lens.cpu().tolist()):
        assert torch.equal(logits[row, :seq_len], ref_logits[row, :seq_len]), f"split C4 paged logits mismatch pages={pages} row={row}"
    assert torch.equal(indices, ref_indices), f"split C4 sparse indices mismatch pages={pages}"
    assert torch.equal(fused_indices, indices), f"fused C4 paged mqa with topk indices mismatch pages={pages}"


def _run_correctness(device):
    _check_rotate(device)
    _check_indexer_store(device)
    _check_flash_store(device)
    _check_act_quant(device)
    _check_topk(device)
    _check_c4_indexer_split_chain(device)
    print("correctness: passed")


def _run_benchmarks(device, repeats, warmup):
    torch.manual_seed(21)
    rows = 4096
    rotate_x = (torch.randn(rows, 128, device=device, dtype=torch.bfloat16) * 0.2).contiguous()
    rotate_ref = rotate_x.clone()
    rotate_native = rotate_x.clone()
    _bench(
        "rotate_128",
        lambda: _infinicore.deepseek_v4_indexer_rotate_naive_(_as_core(rotate_ref.copy_(rotate_x)), True),
        lambda: _infinicore.deepseek_v4_indexer_rotate_128_kernel_(_as_core(rotate_native.copy_(rotate_x)), True),
        repeats,
        warmup,
    )

    page_size = 64
    tokens = 4096
    blocks = (tokens + page_size - 1) // page_size
    indices = torch.arange(tokens, device=device, dtype=torch.int32)
    store_x = (torch.randn(tokens, 128, device=device, dtype=torch.bfloat16) * 0.3).contiguous()
    ref_cache = torch.empty(blocks, _indexer_page_bytes(page_size), device=device, dtype=torch.uint8)
    out_cache = torch.empty_like(ref_cache)
    _bench(
        "indexer_raw_cache_store",
        lambda: _infinicore.deepseek_v4_store_indexer_raw_cache_naive_(_as_core(store_x), _as_core(ref_cache.zero_()), _as_core(indices), page_size),
        lambda: _infinicore.deepseek_v4_store_indexer_raw_cache_kernel_(_as_core(store_x), _as_core(out_cache.zero_()), _as_core(indices), page_size),
        repeats,
        warmup,
    )

    flash_tokens = 1024
    flash_blocks = (flash_tokens + page_size - 1) // page_size
    flash_indices = torch.arange(flash_tokens, device=device, dtype=torch.int64)
    flash_x = (torch.randn(flash_tokens, 512, device=device, dtype=torch.bfloat16) * 0.25).contiguous()
    flash_ref_cache = torch.empty(flash_blocks, _flash_page_bytes(page_size), device=device, dtype=torch.uint8)
    flash_out_cache = torch.empty_like(flash_ref_cache)
    _bench(
        "flashmla_raw_cache_store",
        lambda: _infinicore.deepseek_v4_store_flashmla_raw_cache_naive_(_as_core(flash_x), _as_core(flash_ref_cache.zero_()), _as_core(flash_indices), page_size),
        lambda: _infinicore.deepseek_v4_store_flashmla_raw_cache_kernel_(_as_core(flash_x), _as_core(flash_out_cache.zero_()), _as_core(flash_indices), page_size),
        repeats,
        warmup,
    )

    batch, heads = 512, 32
    q = (torch.randn(batch, heads, 128, device=device, dtype=torch.bfloat16) * 0.2).contiguous()
    weights = torch.randn(batch, heads, device=device, dtype=torch.bfloat16).contiguous()
    ref_fp8 = torch.empty(batch, heads, 128, device=device, dtype=torch.float8_e4m3fn)
    ref_scale = torch.empty(batch, heads, 1, device=device, dtype=torch.float32)
    ref_fused = torch.empty(batch, heads, device=device, dtype=torch.float32)
    out_fp8 = torch.empty_like(ref_fp8)
    out_scale = torch.empty_like(ref_scale)
    out_fused = torch.empty_like(ref_fused)

    def act_ref():
        q_fp8, q_scale, fused = _act_quant_ref(q, weights, 1.0)
        ref_fp8.copy_(q_fp8)
        ref_scale.copy_(q_scale)
        ref_fused.copy_(fused)

    def act_native():
        _infinicore.deepseek_v4_c4_act_quant_fused_scale_kernel_(
            _as_core(q), _as_core(weights), _as_core(out_fp8), _as_core(out_scale), _as_core(out_fused), 1.0
        )

    _bench("act_quant_fused_scale", act_ref, act_native, repeats, warmup)

    batch, max_seq_len = 1024, 384
    scores = torch.randn(batch, max_seq_len, device=device, dtype=torch.float32).contiguous()
    seq_lens = torch.randint(1, max_seq_len + 1, (batch,), device=device, dtype=torch.int32)
    pages = (max_seq_len + page_size - 1) // page_size
    page_table = torch.arange(batch * pages, device=device, dtype=torch.int32).reshape(batch, pages).contiguous()
    topk_ref = torch.empty(batch, TOPK, device=device, dtype=torch.int32)
    topk_out = torch.empty_like(topk_ref)

    def topk_ref_fn():
        topk_ref.copy_(_topk_transform_ref(scores, seq_lens, page_table, page_size))

    def topk_native_fn():
        _infinicore.deepseek_v4_topk_transform_512_kernel_(
            _as_core(scores), _as_core(seq_lens), _as_core(page_table), _as_core(topk_out), page_size
        )

    _bench("topk_transform_512", topk_ref_fn, topk_native_fn, repeats, warmup)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--no-bench", action="store_true")
    args = parser.parse_args()

    device = "cuda"
    _run_correctness(device)
    if not args.no_bench:
        _run_benchmarks(device, args.repeats, args.warmup)


if __name__ == "__main__":
    main()
