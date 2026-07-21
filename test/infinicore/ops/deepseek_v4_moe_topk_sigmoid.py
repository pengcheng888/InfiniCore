import argparse

import infinicore
import sgl_kernel
import torch


def _device(args):
    return "cuda" if args.hygon or args.nvidia else "cuda"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    args = parser.parse_args()
    device = _device(args)

    torch.manual_seed(0)
    gating_output = torch.randn(16, 64, dtype=torch.float32, device=device)
    correction_bias = torch.randn(64, dtype=torch.float32, device=device)
    topk = 8

    ref_weights = torch.empty(16, topk, dtype=torch.float32, device=device)
    ref_indices = torch.empty(16, topk, dtype=torch.int32, device=device)
    sgl_kernel.topk_sigmoid(ref_weights, ref_indices, gating_output, True, correction_bias)
    torch.cuda.synchronize()

    out_weights_t = torch.empty_like(ref_weights)
    out_indices_t = torch.empty_like(ref_indices)
    infinicore.deepseek_v4_moe_topk_sigmoid_(
        infinicore.from_torch(out_weights_t),
        infinicore.from_torch(out_indices_t),
        infinicore.from_torch(gating_output),
        True,
        infinicore.from_torch(correction_bias),
    )
    infinicore.sync_stream()

    assert torch.allclose(out_weights_t, ref_weights, atol=1e-5, rtol=1e-5)
    assert torch.equal(out_indices_t, ref_indices)
    print("DeepseekV4MoeTopkSigmoid: passed")


if __name__ == "__main__":
    main()
