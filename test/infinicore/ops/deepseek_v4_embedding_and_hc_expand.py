import argparse

import infinicore
import torch
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _as_core(tensor):
    return infinicore.from_torch(tensor)


def _copy_to_torch(core_tensor, like):
    out = torch.empty_like(like)
    _as_core(out).copy_(core_tensor)
    infinicore.sync_stream()
    return out


def _reference(input_ids, weight, hc_mult):
    hidden = weight.shape[1]
    embedded = weight.index_select(0, input_ids.reshape(-1))
    expanded = embedded.view(-1, 1, hidden).expand(-1, hc_mult, -1).contiguous()
    return expanded.view(*input_ids.shape, hc_mult, hidden)


def _run_case(device, input_shape, vocab_size, hidden_size, input_dtype, weight_dtype, hc_mult):
    seed = 31 + vocab_size + hidden_size + len(input_shape)
    torch.manual_seed(seed)
    input_ids = torch.randint(
        0,
        vocab_size,
        input_shape,
        device=device,
        dtype=input_dtype,
    )
    weight = torch.randn((vocab_size, hidden_size), device=device, dtype=weight_dtype)
    ref = _reference(input_ids, weight, hc_mult)

    input_core = _as_core(input_ids)
    weight_core = _as_core(weight)

    ret = Tensor(_infinicore.deepseek_v4_embedding_and_hc_expand(input_core._underlying, weight_core._underlying, hc_mult))
    infinicore.sync_stream()
    assert ret.shape == list(ref.shape)
    assert ret.dtype == weight_core.dtype
    assert torch.equal(_copy_to_torch(ret, ref), ref)

    ret_kernel = Tensor(_infinicore.deepseek_v4_embedding_and_hc_expand_kernel(input_core._underlying, weight_core._underlying, hc_mult))
    infinicore.sync_stream()
    assert ret_kernel.shape == list(ref.shape)
    assert ret_kernel.dtype == weight_core.dtype
    assert torch.equal(_copy_to_torch(ret_kernel, ref), ref)

    ret_naive = Tensor(_infinicore.deepseek_v4_embedding_and_hc_expand_naive(input_core._underlying, weight_core._underlying, hc_mult))
    infinicore.sync_stream()
    assert ret_naive.shape == list(ref.shape)
    assert ret_naive.dtype == weight_core.dtype
    assert torch.equal(_copy_to_torch(ret_naive, ref), ref)

    out = torch.empty_like(ref)
    _infinicore.deepseek_v4_embedding_and_hc_expand_(
        _as_core(out)._underlying,
        input_core._underlying,
        weight_core._underlying,
        hc_mult,
    )
    infinicore.sync_stream()
    assert torch.equal(out, ref)

    out_kernel = torch.empty_like(ref)
    _infinicore.deepseek_v4_embedding_and_hc_expand_kernel_(
        _as_core(out_kernel)._underlying,
        input_core._underlying,
        weight_core._underlying,
        hc_mult,
    )
    infinicore.sync_stream()
    assert torch.equal(out_kernel, ref)

    out_naive = torch.empty_like(ref)
    _infinicore.deepseek_v4_embedding_and_hc_expand_naive_(
        _as_core(out_naive)._underlying,
        input_core._underlying,
        weight_core._underlying,
        hc_mult,
    )
    infinicore.sync_stream()
    assert torch.equal(out_naive, ref)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--hc-mult", type=int, default=4)
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if (args.hygon or args.nvidia) and device == "cpu":
        print("DeepseekV4EmbeddingAndHcExpand: accelerator unavailable, running CPU correctness fallback")

    cases = [
        ((1,), 128, 16, torch.int64, torch.float32),
        ((7,), 32000, 128, torch.int32, torch.float16),
        ((2, 5), 32000, 512, torch.int64, torch.bfloat16),
        ((11,), 129280, 4096, torch.int64, torch.bfloat16),
    ]
    for case in cases:
        _run_case(device, *case, args.hc_mult)

    print("DeepseekV4EmbeddingAndHcExpand: passed")


if __name__ == "__main__":
    main()
