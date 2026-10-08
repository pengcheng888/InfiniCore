import math

import torch

import infinicore

DEVICE = "cuda"
DTYPE = torch.bfloat16


def vendor_ref(name):
    if name == "argsort_bincount_with_inv_pos":
        from vllm_iluvatar.custom_kernels.argsort_bincount_with_inv_pos import (
            argsort_bincount_with_inv_pos,
        )

        return argsort_bincount_with_inv_pos
    if name == "concat_and_cache_mla":
        from vllm_iluvatar.custom_kernels.concat_and_cache_mla import (
            concat_and_cache_mla,
        )

        return concat_and_cache_mla
    if name == "concat_mla_q":
        from vllm_iluvatar.custom_kernels.concat_mla_q import concat_mla_q

        return concat_mla_q
    if name == "dynamic_scaled_int8_quant":
        from vllm_iluvatar.custom_kernels.dynamic_scaled_int8_quant import (
            dynamic_scaled_int8_quant,
        )

        return dynamic_scaled_int8_quant
    if name == "grouped_topk":
        from vllm_iluvatar.custom_kernels.grouped_topk import grouped_topk

        return grouped_topk
    if name == "moe_sum":
        from vllm_iluvatar.custom_kernels.moe_sum import moe_sum

        return moe_sum
    if name == "rotary_embedding":
        from vllm_iluvatar.custom_kernels.rotary_embedding import rotary_embedding

        return rotary_embedding
    if name == "scaled_mm_w4a8":
        from vllm_iluvatar.custom_kernels.scaled_mm_w4a8 import scaled_mm_w4a8

        return scaled_mm_w4a8
    if name == "topk_sigmoid":
        from vllm_iluvatar.custom_kernels.topk_sigmoid import topk_sigmoid

        return topk_sigmoid
    if name == "topk_softmax":
        from vllm_iluvatar.custom_kernels.topk_softmax import topk_softmax

        return topk_softmax
    if name == "w4a8_group_gemm":
        from vllm_iluvatar.custom_kernels.w4a8_group_gemm import w4a8_group_gemm

        return w4a8_group_gemm
    if name == "w8a8_group_gemm":
        from vllm_iluvatar.custom_kernels.w8a8_group_gemm import w8a8_group_gemm

        return w8a8_group_gemm
    if name == "w16a16_group_gemm":
        from vllm_iluvatar.custom_kernels.w16a16_group_gemm import w16a16_group_gemm

        return w16a16_group_gemm
    raise ValueError(name)


def ic(tensor):
    return infinicore.from_torch(tensor)


def close(actual, expected, *, rtol=1e-2, atol=1e-2):
    infinicore.sync_device()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def test_add_rms_norm_inplace():
    x = torch.randn(4, 128, device=DEVICE, dtype=DTYPE)
    residual = torch.randn_like(x)
    weight = torch.randn(128, device=DEVICE, dtype=DTYPE)
    expected_residual = x + residual
    expected = (
        expected_residual.float()
        * torch.rsqrt(expected_residual.float().pow(2).mean(-1, keepdim=True) + 1e-5)
        * weight.float()
    ).to(DTYPE)
    infinicore.add_rms_norm_inplace(ic(x), ic(residual), ic(weight))
    close(residual, expected_residual)
    close(x, expected)


def test_mla_concat_and_cache():
    concat_mla_q = vendor_ref("concat_mla_q")
    concat_and_cache_mla = vendor_ref("concat_and_cache_mla")
    ql = torch.randn(2, 8, 512, device=DEVICE, dtype=DTYPE)
    qpe = torch.randn(2, 8, 64, device=DEVICE, dtype=DTYPE)
    q_ref = torch.empty(2, 8, 576, device=DEVICE, dtype=DTYPE)
    q_actual = torch.empty_like(q_ref)
    concat_mla_q(ql, qpe, q_ref)
    infinicore.concat_mla_q(ic(ql), ic(qpe), out=ic(q_actual))
    close(q_actual, q_ref)

    kv = torch.randn(2, 512, device=DEVICE, dtype=DTYPE)
    kpe = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)
    slots = torch.tensor([1, 66], device=DEVICE, dtype=torch.int64)
    scale = torch.ones(1, device=DEVICE, dtype=torch.float32)
    cache_ref = torch.zeros(2, 64, 576, device=DEVICE, dtype=DTYPE)
    cache_out = cache_ref.clone()
    concat_and_cache_mla(kv, kpe, cache_ref, slots, "auto", scale)
    infinicore.concat_and_cache_mla(
        ic(kv), ic(kpe), ic(cache_out), ic(slots), "auto", ic(scale)
    )
    close(cache_out, cache_ref)



def test_mla_concat_and_cache_block16_slots():
    kv = torch.randn(9, 512, device=DEVICE, dtype=DTYPE)
    kpe = torch.randn(9, 64, device=DEVICE, dtype=DTYPE)
    slots = torch.arange(9, device=DEVICE, dtype=torch.int64)
    scale = torch.ones(1, device=DEVICE, dtype=torch.float32)
    cache = torch.zeros(4, 16, 576, device=DEVICE, dtype=DTYPE)
    infinicore.concat_and_cache_mla(
        ic(kv), ic(kpe), ic(cache), ic(slots), "auto", ic(scale)
    )
    expected = torch.zeros_like(cache)
    expected[0, :9, :512] = kv
    expected[0, :9, 512:] = kpe
    close(cache, expected, rtol=0, atol=0)


def test_mla_int8_cache():
    kv = torch.randint(-100, 100, (2, 512), device=DEVICE, dtype=torch.int8)
    kpe = torch.randint(-100, 100, (2, 64), device=DEVICE, dtype=torch.int8)
    kv_scale = torch.rand(2, device=DEVICE)
    kpe_scale = torch.rand(2, device=DEVICE)
    slots = torch.tensor([1, 66], device=DEVICE, dtype=torch.int64)
    cache = torch.zeros(2, 64, 576, device=DEVICE, dtype=torch.int8)
    cache_scale = torch.zeros(2, 64, 2, device=DEVICE)
    infinicore.concat_and_cache_mla_int8(
        ic(kv),
        ic(kv_scale),
        ic(kpe),
        ic(kpe_scale),
        ic(cache),
        ic(cache_scale),
        ic(slots),
    )
    infinicore.sync_device()
    flat_cache = cache.view(-1, 576)
    flat_scale = cache_scale.view(-1, 2)
    torch.testing.assert_close(flat_cache[slots], torch.cat((kv, kpe), dim=-1))
    torch.testing.assert_close(
        flat_scale[slots], torch.stack((kv_scale, kpe_scale), dim=-1)
    )


def test_dynamic_int8_quant():
    dynamic_scaled_int8_quant = vendor_ref("dynamic_scaled_int8_quant")
    x = torch.randn(8, 128, device=DEVICE, dtype=DTYPE)
    ref = torch.empty_like(x, dtype=torch.int8)
    ref_scale = torch.empty(8, device=DEVICE)
    dynamic_scaled_int8_quant(ref, x, ref_scale)
    out = torch.empty_like(ref)
    scale = torch.empty_like(ref_scale)
    infinicore.dynamic_scaled_int8_quant(ic(x), ic(scale), out=ic(out))
    close(out, ref, rtol=0, atol=0)
    close(scale, ref_scale, rtol=1e-5, atol=1e-6)


def test_allocating_wrappers():
    concat_mla_q = vendor_ref("concat_mla_q")
    dynamic_scaled_int8_quant = vendor_ref("dynamic_scaled_int8_quant")
    scaled_mm_w4a8 = vendor_ref("scaled_mm_w4a8")
    ql = torch.randn(2, 8, 512, device=DEVICE, dtype=DTYPE)
    qpe = torch.randn(2, 8, 64, device=DEVICE, dtype=DTYPE)
    q_ref = torch.empty(2, 8, 576, device=DEVICE, dtype=DTYPE)
    concat_mla_q(ql, qpe, q_ref)
    q_result = infinicore.concat_mla_q(ic(ql), ic(qpe))
    q_actual = torch.empty_like(q_ref)
    ic(q_actual).copy_(q_result)
    close(q_actual, q_ref)

    x = torch.randn(8, 128, device=DEVICE, dtype=DTYPE)
    quant_ref = torch.empty_like(x, dtype=torch.int8)
    scale_ref = torch.empty(8, device=DEVICE)
    dynamic_scaled_int8_quant(quant_ref, x, scale_ref)
    scale_actual = torch.empty_like(scale_ref)
    quant_result = infinicore.dynamic_scaled_int8_quant(ic(x), ic(scale_actual))
    quant_actual = torch.empty_like(quant_ref)
    ic(quant_actual).copy_(quant_result)
    close(quant_actual, quant_ref, rtol=0, atol=0)
    close(scale_actual, scale_ref, rtol=1e-5, atol=1e-6)

    m, n, k = 16, 128, 128
    a = torch.randint(-8, 8, (m, k), device=DEVICE, dtype=torch.int8)
    b4 = torch.randint(-128, 127, (k, n // 2), device=DEVICE, dtype=torch.int8)
    b8 = torch.randint(-8, 8, (n, k), device=DEVICE, dtype=torch.int8)
    a_scale = torch.rand(m, 1, device=DEVICE)
    b_scale = torch.rand(n, 1, device=DEVICE)
    ref4 = torch.empty(m, n, device=DEVICE, dtype=torch.float16)
    scaled_mm_w4a8(ref4, a, b4, a_scale, b_scale)
    result4 = infinicore.scaled_mm_w4a8(
        ic(a), ic(b4), ic(a_scale), ic(b_scale), trans_weight=False
    )
    actual4 = torch.empty_like(ref4)
    ic(actual4).copy_(result4)
    close(actual4, ref4)

    result8 = infinicore.scaled_mm_w8a8(
        ic(a), ic(b8), ic(a_scale), ic(b_scale), trans_weight=True
    )
    actual8 = torch.empty_like(ref4)
    ic(actual8).copy_(result8)
    expected8 = ((a.float() * a_scale) @ (b8.float() * b_scale).transpose(0, 1)).half()
    close(actual8, expected8, rtol=5e-2, atol=1.0)


def test_rotary_embedding():
    rotary_embedding = vendor_ref("rotary_embedding")
    positions = torch.tensor([0, 3], device=DEVICE, dtype=torch.int64)
    query_ref = torch.randn(2, 8, 128, device=DEVICE, dtype=DTYPE)
    key_ref = torch.randn(2, 2, 128, device=DEVICE, dtype=DTYPE)
    query_out, key_out = query_ref.clone(), key_ref.clone()
    cache = torch.randn(16, 128, device=DEVICE, dtype=DTYPE)
    rotary_embedding(
        positions, query_ref.view(2, -1), key_ref.view(2, -1), 128, cache, True
    )
    infinicore.fused_rotary_embedding_(
        ic(query_out), ic(key_out), ic(positions), 128, ic(cache), True
    )
    close(query_out, query_ref)
    close(key_out, key_ref)


def test_routing_ops():
    grouped_topk = vendor_ref("grouped_topk")
    topk_softmax = vendor_ref("topk_softmax")
    topk_sigmoid = vendor_ref("topk_sigmoid")
    scores = torch.randn(4, 64, device=DEVICE, dtype=DTYPE)
    bias = torch.randn(64, device=DEVICE, dtype=DTYPE)
    ref_w, ref_i = grouped_topk(scores, 8, 4, 4, True, 1.25, bias, 1)
    actual_w = torch.empty_like(ref_w)
    actual_i = torch.empty_like(ref_i)
    infinicore.grouped_topk_vendor(
        ic(scores),
        8,
        4,
        4,
        True,
        1.25,
        ic(bias),
        "sigmoid",
        out=(ic(actual_w), ic(actual_i)),
    )
    close(actual_w, ref_w)
    close(actual_i, ref_i, rtol=0, atol=0)

    for reference, candidate in (
        (topk_softmax, infinicore.moe_topk_softmax_vendor),
        (topk_sigmoid, infinicore.moe_topk_sigmoid_vendor),
    ):
        ref_w = torch.empty(4, 4, device=DEVICE)
        ref_i = torch.empty(4, 4, device=DEVICE, dtype=torch.int32)
        ref_t = torch.empty_like(ref_i)
        reference(ref_w, ref_i, ref_t, scores, True, bias)
        actual_w = torch.empty_like(ref_w)
        actual_i = torch.empty_like(ref_i)
        actual_t = torch.empty_like(ref_t)
        candidate(
            ic(scores),
            4,
            renormalize=True,
            correction_bias=ic(bias),
            out=(ic(actual_w), ic(actual_i), ic(actual_t)),
        )
        close(actual_w, ref_w)
        close(actual_i, ref_i, rtol=0, atol=0)
        assert torch.all((actual_t >= 0) & (actual_t < 64))


def test_moe_data_movement_and_sum():
    argsort_bincount_with_inv_pos = vendor_ref("argsort_bincount_with_inv_pos")
    moe_sum = vendor_ref("moe_sum")
    topk_ids = torch.tensor(
        [[3, 1], [0, 3], [2, 1], [0, 2]], device=DEVICE, dtype=torch.int32
    )
    ref_tpe, ref_sorted, ref_inv = argsort_bincount_with_inv_pos(topk_ids, 4)
    tpe = torch.empty(4, device=DEVICE, dtype=torch.int32)
    sorted_ids = torch.empty(8, device=DEVICE, dtype=torch.int32)
    inv = torch.empty_like(sorted_ids)
    infinicore.moe_argsort_bincount_with_inv_pos_(
        ic(tpe), ic(sorted_ids), ic(inv), ic(topk_ids), 4
    )
    close(tpe, ref_tpe.to(DEVICE), rtol=0, atol=0)
    close(sorted_ids, ref_sorted, rtol=0, atol=0)
    close(inv, ref_inv, rtol=0, atol=0)

    hidden = torch.randn(4, 128, device=DEVICE, dtype=DTYPE)
    expanded = torch.empty(8, 128, device=DEVICE, dtype=DTYPE)
    infinicore.moe_expand_input_with_inv_pos_(
        ic(expanded), None, ic(hidden), ic(inv), 2
    )
    close(expanded, hidden.repeat_interleave(2, dim=0)[ref_sorted])

    gated = torch.randn(8, 256, device=DEVICE, dtype=DTYPE)
    activated = torch.empty(8, 128, device=DEVICE, dtype=DTYPE)
    infinicore.moe_silu_and_mul_quant_(ic(activated), None, ic(gated))
    expected = torch.nn.functional.silu(gated[:, :128]) * gated[:, 128:]
    close(activated, expected)

    expert_out = torch.randn(4, 2, 128, device=DEVICE, dtype=DTYPE)
    weights = torch.rand(4, 2, device=DEVICE)
    residual = torch.randn(4, 128, device=DEVICE, dtype=DTYPE)
    ref = torch.empty_like(residual)
    out = torch.empty_like(residual)
    moe_sum(expert_out, ref, weights, residual, 1.25, 0.5)
    infinicore.moe_sum_vendor_(
        ic(out), ic(expert_out), ic(weights), ic(residual), 1.25, 0.5
    )
    close(out, ref)


def test_scaled_mm():
    scaled_mm_w4a8 = vendor_ref("scaled_mm_w4a8")
    m, n, k = 16, 128, 128
    a = torch.randint(-8, 8, (m, k), device=DEVICE, dtype=torch.int8)
    b4 = torch.randint(-128, 127, (k, n // 2), device=DEVICE, dtype=torch.int8)
    a_scale = torch.rand(m, 1, device=DEVICE)
    b_scale = torch.rand(n, 1, device=DEVICE)
    ref = torch.empty(m, n, device=DEVICE, dtype=torch.float16)
    scaled_mm_w4a8(ref, a, b4, a_scale, b_scale)
    out = torch.empty_like(ref)
    infinicore.scaled_mm_w4a8(
        ic(a), ic(b4), ic(a_scale), ic(b_scale), trans_weight=False, out=ic(out)
    )
    close(out, ref)

    b8 = torch.randint(-8, 8, (n, k), device=DEVICE, dtype=torch.int8)
    out8 = torch.empty_like(ref)
    infinicore.scaled_mm_w8a8(
        ic(a), ic(b8), ic(a_scale), ic(b_scale), trans_weight=True, out=ic(out8)
    )
    expected8 = ((a.float() * a_scale) @ (b8.float() * b_scale).transpose(0, 1)).half()
    close(out8, expected8, rtol=5e-2, atol=1.0)


def test_group_gemm():
    w4a8_group_gemm = vendor_ref("w4a8_group_gemm")
    w8a8_group_gemm = vendor_ref("w8a8_group_gemm")
    w16a16_group_gemm = vendor_ref("w16a16_group_gemm")
    experts, rows, n, k = 2, 16, 128, 128
    tokens = torch.tensor([8, 8], dtype=torch.int32)

    x16 = torch.randn(rows, k, device=DEVICE, dtype=DTYPE)
    w16 = torch.randn(experts, n, k, device=DEVICE, dtype=DTYPE)
    ref16, out16 = (
        torch.empty(rows, n, device=DEVICE, dtype=DTYPE),
        torch.empty(rows, n, device=DEVICE, dtype=DTYPE),
    )
    w16a16_group_gemm(ref16, x16, w16, tokens)
    infinicore.w16a16_group_gemm_(ic(out16), ic(x16), ic(w16), ic(tokens))
    close(out16, ref16)

    x8 = torch.randint(-8, 8, (rows, k), device=DEVICE, dtype=torch.int8)
    w8 = torch.randint(-8, 8, (experts, n, k), device=DEVICE, dtype=torch.int8)
    xs = torch.rand(rows, 1, device=DEVICE)
    ws = torch.rand(experts, n, 1, device=DEVICE)
    ref8, out8 = torch.empty_like(ref16), torch.empty_like(ref16)
    w8a8_group_gemm(ref8, x8, w8, xs, ws, tokens)
    infinicore.w8a8_group_gemm_(ic(out8), ic(x8), ic(w8), ic(xs), ic(ws), ic(tokens))
    close(out8, ref8)

    w4 = torch.randint(-128, 127, (experts, n, k // 2), device=DEVICE, dtype=torch.int8)
    ref4, out4 = torch.empty_like(ref16), torch.empty_like(ref16)
    w4a8_group_gemm(ref4, x8, w4, xs, ws, tokens)
    infinicore.w4a8_group_gemm_(ic(out4), ic(x8), ic(w4), ic(xs), ic(ws), ic(tokens))
    close(out4, ref4)


def test_vendor_graph_replay():
    kv = torch.randn(2, 512, device=DEVICE, dtype=DTYPE)
    kpe = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)
    slots = torch.tensor([1, 66], device=DEVICE, dtype=torch.int64)
    scale = torch.ones(1, device=DEVICE, dtype=torch.float32)
    cache_ref = torch.zeros(2, 64, 576, device=DEVICE, dtype=DTYPE)
    cache_out = cache_ref.clone()
    cache_ref.view(-1, 576)[slots, :512] = kv
    cache_ref.view(-1, 576)[slots, 512:] = kpe

    torch.cuda.synchronize()
    infinicore.start_graph_recording()
    infinicore.concat_and_cache_mla(
        ic(kv), ic(kpe), ic(cache_out), ic(slots), "auto", ic(scale)
    )
    graph = infinicore.stop_graph_recording()
    graph.run()
    close(cache_out, cache_ref)


def test_paged_attention_mla_smoke():
    query = torch.randn(1, 8, 576, device=DEVICE, dtype=DTYPE)
    cache = torch.randn(1, 16, 576, device=DEVICE, dtype=DTYPE)
    blocks = torch.tensor([[0]], device=DEVICE, dtype=torch.int32)
    lens = torch.tensor([16], device=DEVICE, dtype=torch.int32)
    out = torch.empty(1, 8, 512, device=DEVICE, dtype=DTYPE)
    infinicore.paged_attention_mla_(
        ic(out), ic(query), ic(cache), 576**-0.5, ic(blocks), ic(lens), 16
    )
    infinicore.sync_device()
    assert torch.isfinite(out).all()


def _check_flash_mla_with_kvcache_dense_decode(block_size):
    query = torch.randn(2, 1, 8, 576, device=DEVICE, dtype=DTYPE)
    cache = torch.randn(4, block_size, 1, 576, device=DEVICE, dtype=DTYPE)
    blocks = torch.tensor([[0, 1], [2, 3]], device=DEVICE, dtype=torch.int32)
    lens = torch.tensor(
        [min(10, block_size), block_size + 6], device=DEVICE, dtype=torch.int32
    )

    sched_meta = infinicore.FlashMLASchedMeta()
    out, lse = infinicore.flash_mla_with_kvcache(
        ic(query),
        ic(cache),
        ic(blocks),
        ic(lens),
        512,
        sched_meta,
        softmax_scale=576**-0.5,
    )
    out_torch = torch.empty(2, 1, 8, 512, device=DEVICE, dtype=DTYPE)
    lse_torch = torch.empty(2, 8, 1, device=DEVICE, dtype=torch.float32)
    ic(out_torch).copy_(out)
    ic(lse_torch).copy_(lse)

    flat_cache = cache.squeeze(2).reshape(-1, 576)
    blocks_cpu = blocks.cpu()
    lens_cpu = lens.cpu()
    ref_out = []
    ref_lse = []
    for batch_idx in range(query.shape[0]):
        gather = []
        for token_idx in range(int(lens_cpu[batch_idx])):
            block_idx = int(blocks_cpu[batch_idx, token_idx // block_size])
            gather.append(block_idx * block_size + token_idx % block_size)
        kv = flat_cache[torch.tensor(gather, device=DEVICE)].float()
        scores = (query[batch_idx, 0].float() @ kv.t()) * (576**-0.5)
        ref_out.append((scores.softmax(dim=-1) @ kv[:, :512]).to(DTYPE))
        # The Iluvatar FlashMLA kernel reports LSE in base 2.
        ref_lse.append(scores.logsumexp(dim=-1) * math.log2(math.e))
    close(out_torch, torch.stack(ref_out).unsqueeze(1))
    close(lse_torch, torch.stack(ref_lse).unsqueeze(-1), rtol=1e-4, atol=1e-4)

    graph_cache = cache.clone()
    graph_sched_meta = infinicore.get_mla_metadata(ic(lens), 8, 1)
    infinicore.start_graph_recording()
    graph_out, graph_lse = infinicore.flash_mla_with_kvcache(
        ic(query),
        ic(graph_cache),
        ic(blocks),
        ic(lens),
        512,
        graph_sched_meta,
        softmax_scale=576**-0.5,
    )
    graph = infinicore.stop_graph_recording()
    graph.run()

    graph_out_torch = torch.empty_like(out_torch)
    graph_lse_torch = torch.empty_like(lse_torch)
    ic(graph_out_torch).copy_(graph_out)
    ic(graph_lse_torch).copy_(graph_lse)
    close(graph_out_torch, out_torch, rtol=0, atol=0)
    close(graph_lse_torch, lse_torch, rtol=0, atol=0)


def test_flash_mla_with_kvcache_dense_decode_smoke():
    _check_flash_mla_with_kvcache_dense_decode(16)


def _flash_mla_sparse_reference(query, flat_cache, indices, topk_lengths, sink, scale):
    ref = torch.empty_like(query[..., :512])
    for token_idx in range(query.shape[0]):
        valid = indices[token_idx, 0, : int(topk_lengths[token_idx])].long()
        kv = flat_cache[valid].float()
        for head_idx in range(query.shape[2]):
            logits = (query[token_idx, 0, head_idx].float() @ kv.t()) * scale
            max_logit = max(float(logits.max()), float(sink[head_idx]))
            weights = torch.exp(logits - max_logit)
            denominator = weights.sum() + math.exp(float(sink[head_idx]) - max_logit)
            ref[token_idx, 0, head_idx] = ((weights @ kv) / denominator).to(ref.dtype)
    return ref


def _check_flash_mla_sparse_decode():
    tokens, heads, head_dim, topk_width = 2, 8, 512, 16
    scale = head_dim**-0.5
    query = torch.randn(tokens, 1, heads, head_dim, device=DEVICE, dtype=DTYPE)
    cache = torch.randn(4, 16, 1, head_dim, device=DEVICE, dtype=DTYPE)
    flat_cache = cache.reshape(-1, head_dim)
    indices = torch.tensor(
        [[3, 4, 5, 6, 7, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1, -1],
         [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, -1, -1, -1, -1]],
        device=DEVICE,
        dtype=torch.int32,
    ).view(tokens, 1, topk_width)
    topk_lengths = torch.tensor([8, 12], device=DEVICE, dtype=torch.int32)
    sink = torch.randn(heads, device=DEVICE, dtype=torch.float32)

    sched_meta = infinicore.get_mla_metadata(
        ic(topk_lengths), heads, 1, num_heads_q=heads, is_fp8_kvcache=False, topk=topk_width
    )
    out, lse = infinicore.flash_mla_with_kvcache(
        ic(query),
        ic(cache),
        None,
        None,
        512,
        sched_meta,
        softmax_scale=scale,
        indices=ic(indices),
        attn_sink=ic(sink),
        topk_length=ic(topk_lengths),
    )
    out_torch = torch.empty(tokens, 1, heads, 512, device=DEVICE, dtype=DTYPE)
    lse_torch = torch.empty(tokens, heads, 1, device=DEVICE, dtype=torch.float32)
    ic(out_torch).copy_(out)
    ic(lse_torch).copy_(lse)
    close(out_torch, _flash_mla_sparse_reference(query, flat_cache, indices, topk_lengths, sink, scale))
    assert torch.count_nonzero(lse_torch) == 0

    # Replay must consume the graph-bound indices/length buffers, not the
    # values captured at recording time.
    indices_second = torch.tensor(
        [[0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
         [32, 33, 34, 35, 36, 37, 38, 39, -1, -1, -1, -1, -1, -1, -1, -1]],
        device=DEVICE,
        dtype=torch.int32,
    ).view(tokens, 1, topk_width)
    lengths_second = torch.tensor([5, 8], device=DEVICE, dtype=torch.int32)
    indices.copy_(indices_second)
    topk_lengths.copy_(lengths_second)

    infinicore.start_graph_recording()
    graph_out, _ = infinicore.flash_mla_with_kvcache(
        ic(query),
        ic(cache),
        None,
        None,
        512,
        sched_meta,
        softmax_scale=scale,
        indices=ic(indices),
        attn_sink=ic(sink),
        topk_length=ic(topk_lengths),
    )
    graph = infinicore.stop_graph_recording()
    graph.run()

    graph_out_torch = torch.empty_like(out_torch)
    ic(graph_out_torch).copy_(graph_out)
    close(
        graph_out_torch,
        _flash_mla_sparse_reference(query, flat_cache, indices, topk_lengths, sink, scale),
    )


def test_flash_mla_sparse_decode_smoke():
    _check_flash_mla_sparse_decode()


def test_flash_mla_sparse_rejects_unsupported_options():
    query = torch.randn(1, 1, 2, 512, device=DEVICE, dtype=DTYPE)
    cache = torch.randn(2, 16, 1, 512, device=DEVICE, dtype=DTYPE)
    indices = torch.arange(8, device=DEVICE, dtype=torch.int32).view(1, 1, 8)
    topk_lengths = torch.tensor([8], device=DEVICE, dtype=torch.int32)
    extra = torch.randn(1, 16, 1, 512, device=DEVICE, dtype=DTYPE)
    sched_meta = infinicore.get_mla_metadata(
        ic(topk_lengths), 2, 1, num_heads_q=2, is_fp8_kvcache=False, topk=8
    )
    for kwargs in (
        {"is_fp8_kvcache": True},
        {"extra_k_cache": ic(extra)},
    ):
        try:
            infinicore.flash_mla_with_kvcache(
                ic(query),
                ic(cache),
                None,
                None,
                512,
                sched_meta,
                softmax_scale=512**-0.5,
                indices=ic(indices),
                topk_length=ic(topk_lengths),
                **kwargs,
            )
        except RuntimeError as error:
            assert "ILUVATAR" in str(error) or "Iluvatar" in str(error)
        else:
            raise AssertionError(f"expected sparse option rejection: {kwargs}")


if __name__ == "__main__":
    torch.manual_seed(0)
    for test in (
        test_add_rms_norm_inplace,
        test_mla_concat_and_cache,
        test_mla_concat_and_cache_block16_slots,
        test_mla_int8_cache,
        test_dynamic_int8_quant,
        test_allocating_wrappers,
        test_rotary_embedding,
        test_routing_ops,
        test_moe_data_movement_and_sum,
        test_scaled_mm,
        test_group_gemm,
        test_vendor_graph_replay,
        test_paged_attention_mla_smoke,
        test_flash_mla_with_kvcache_dense_decode_smoke,
        test_flash_mla_sparse_decode_smoke,
        test_flash_mla_sparse_rejects_unsupported_options,
    ):
        test()
        print(f"{test.__name__} ok")
