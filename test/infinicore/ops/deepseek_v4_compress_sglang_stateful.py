import argparse

import infinicore
import torch
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor as CoreTensor


def _as_core(tensor):
    return infinicore.from_torch(tensor)._underlying


def _sync():
    infinicore.sync_stream()
    torch.cuda.synchronize()


def _copy_to_torch(core_tensor, shape, dtype=torch.bfloat16):
    out = torch.empty(shape, device="cuda", dtype=dtype)
    infinicore.from_torch(out).copy_(CoreTensor(core_tensor))
    _sync()
    return out


def _assert_close(name, got, ref, atol=2e-2, rtol=2e-2):
    try:
        torch.testing.assert_close(got, ref, atol=atol, rtol=rtol)
    except AssertionError as exc:
        diff = (got.float() - ref.float()).abs().max().item()
        raise AssertionError(f"{name} mismatch, max_abs={diff}") from exc


def check_c4_sglang_stateful(ape_shape):
    torch.manual_seed(3 + ape_shape[0])
    tokens, dim = 8, 512
    kv_score = torch.randn((tokens, 4 * dim), device="cuda", dtype=torch.bfloat16)
    ape = torch.randn(ape_shape, device="cuda", dtype=torch.bfloat16)
    state_base = torch.zeros((8, 4 * dim), device="cuda", dtype=torch.float32)
    write_loc = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device="cuda", dtype=torch.int32)
    extra_loc = torch.tensor([-1, -1, -1, -1, 0, 0, 0, 0], device="cuda", dtype=torch.int32).reshape(tokens, 1)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int32)

    ref_state = state_base.clone()
    got_state = state_base.clone()
    ref_core = _infinicore.deepseek_v4_c4_compress_stateful_naive(
        _as_core(kv_score),
        _as_core(ape),
        _as_core(ref_state),
        _as_core(write_loc),
        _as_core(extra_loc),
        _as_core(positions),
    )
    got_core = _infinicore.deepseek_v4_c4_compress_sglang_stateful_kernel(
        _as_core(kv_score),
        _as_core(ape),
        _as_core(got_state),
        _as_core(write_loc),
        _as_core(extra_loc),
        _as_core(positions),
    )
    ref = _copy_to_torch(ref_core, (tokens, dim), kv_score.dtype)
    got = _copy_to_torch(got_core, (tokens, dim), kv_score.dtype)
    _assert_close(f"c4 sglang output ape={ape_shape}", got, ref)
    _assert_close(f"c4 sglang state ape={ape_shape}", got_state, ref_state, atol=0, rtol=0)


def check_c4_rejects_legacy_ape_shape():
    torch.manual_seed(13)
    tokens, dim = 8, 512
    kv_score = torch.randn((tokens, 4 * dim), device="cuda", dtype=torch.bfloat16)
    ape = torch.randn((4, 2 * dim), device="cuda", dtype=torch.bfloat16)
    state = torch.zeros((8, 4 * dim), device="cuda", dtype=torch.float32)
    write_loc = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device="cuda", dtype=torch.int32)
    extra_loc = torch.tensor([-1, -1, -1, -1, 0, 0, 0, 0], device="cuda", dtype=torch.int32).reshape(tokens, 1)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int32)

    try:
        _infinicore.deepseek_v4_c4_compress_sglang_stateful_kernel(
            _as_core(kv_score),
            _as_core(ape),
            _as_core(state),
            _as_core(write_loc),
            _as_core(extra_loc),
            _as_core(positions),
        )
    except RuntimeError as exc:
        if "expects ape [8, head_dim]" not in str(exc):
            raise AssertionError(f"unexpected legacy ape error: {exc}") from exc
        return
    raise AssertionError("c4 sglang kernel accepted legacy ape [4, 2 * head_dim]")


def check_c128_sglang_stateful():
    torch.manual_seed(4)
    tokens, dim = 128, 512
    kv_score = torch.randn((tokens, 2 * dim), device="cuda", dtype=torch.bfloat16)
    ape = torch.randn((128, dim), device="cuda", dtype=torch.bfloat16)
    state_base = torch.zeros((128, 2 * dim), device="cuda", dtype=torch.float32)
    write_loc = torch.zeros((tokens,), device="cuda", dtype=torch.int32)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int32)

    ref_state = state_base.clone()
    got_state = state_base.clone()
    ref_core = _infinicore.deepseek_v4_c128_compress_stateful_naive(
        _as_core(kv_score), _as_core(ape), _as_core(ref_state), _as_core(write_loc), _as_core(positions)
    )
    got_core = _infinicore.deepseek_v4_c128_compress_sglang_stateful_kernel(
        _as_core(kv_score), _as_core(ape), _as_core(got_state), _as_core(write_loc), _as_core(positions)
    )
    ref = _copy_to_torch(ref_core, (tokens, dim), kv_score.dtype)
    got = _copy_to_torch(got_core, (tokens, dim), kv_score.dtype)
    _assert_close("c128 sglang output", got, ref)
    _assert_close("c128 sglang state", got_state, ref_state, atol=0, rtol=0)


def check_sglang_stateful_graph():
    torch.manual_seed(5)
    tokens, dim = 8, 512
    kv_score = torch.randn((tokens, 4 * dim), device="cuda", dtype=torch.bfloat16)
    ape = torch.randn((8, dim), device="cuda", dtype=torch.bfloat16)
    state_base = torch.zeros((8, 4 * dim), device="cuda", dtype=torch.float32)
    write_loc = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device="cuda", dtype=torch.int32)
    extra_loc = torch.tensor([-1, -1, -1, -1, 0, 0, 0, 0], device="cuda", dtype=torch.int32).reshape(tokens, 1)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int32)

    ref_state = state_base.clone()
    ref_core = _infinicore.deepseek_v4_c4_compress_stateful_naive(
        _as_core(kv_score), _as_core(ape), _as_core(ref_state), _as_core(write_loc), _as_core(extra_loc), _as_core(positions)
    )
    ref = _copy_to_torch(ref_core, (tokens, dim), kv_score.dtype)

    graph_state = state_base.clone()
    infinicore.start_graph_recording()
    graph_core = _infinicore.deepseek_v4_c4_compress_sglang_stateful_kernel(
        _as_core(kv_score), _as_core(ape), _as_core(graph_state), _as_core(write_loc), _as_core(extra_loc), _as_core(positions)
    )
    graph_obj = infinicore.stop_graph_recording()
    graph_state.copy_(state_base)
    _sync()
    graph_obj.run()
    got = _copy_to_torch(graph_core, (tokens, dim), kv_score.dtype)
    _assert_close("c4 sglang graph output", got, ref)
    _assert_close("c4 sglang graph state", graph_state, ref_state, atol=0, rtol=0)

    tokens = 128
    kv_score = torch.randn((tokens, 2 * dim), device="cuda", dtype=torch.bfloat16)
    ape = torch.randn((128, dim), device="cuda", dtype=torch.bfloat16)
    state_base = torch.zeros((128, 2 * dim), device="cuda", dtype=torch.float32)
    write_loc = torch.zeros((tokens,), device="cuda", dtype=torch.int32)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int32)

    ref_state = state_base.clone()
    ref_core = _infinicore.deepseek_v4_c128_compress_stateful_naive(
        _as_core(kv_score), _as_core(ape), _as_core(ref_state), _as_core(write_loc), _as_core(positions)
    )
    ref = _copy_to_torch(ref_core, (tokens, dim), kv_score.dtype)

    graph_state = state_base.clone()
    infinicore.start_graph_recording()
    graph_core = _infinicore.deepseek_v4_c128_compress_sglang_stateful_kernel(
        _as_core(kv_score), _as_core(ape), _as_core(graph_state), _as_core(write_loc), _as_core(positions)
    )
    graph_obj = infinicore.stop_graph_recording()
    graph_state.copy_(state_base)
    _sync()
    graph_obj.run()
    got = _copy_to_torch(graph_core, (tokens, dim), kv_score.dtype)
    _assert_close("c128 sglang graph output", got, ref)
    _assert_close("c128 sglang graph state", graph_state, ref_state, atol=0, rtol=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.parse_args()

    check_c4_sglang_stateful((8, 512))
    check_c4_rejects_legacy_ape_shape()
    check_c128_sglang_stateful()
    check_sglang_stateful_graph()
    print("deepseek_v4_compress_sglang_stateful correctness ok")


if __name__ == "__main__":
    main()
