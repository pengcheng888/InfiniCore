import argparse

import infinicore
import torch
import vllm._C  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.parse_args()

    torch.manual_seed(3)
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    epsilon = 1e-6
    group_size = 64

    ref_out = torch.empty_like(x, dtype=torch.int8)
    ref_scale = torch.empty((x.numel() // x.shape[-1], x.shape[-1] // group_size), device=x.device, dtype=torch.float32)
    torch.ops._C.rms_norm_per_block_quant(ref_out, x, weight, ref_scale, epsilon, None, None, group_size, False)
    torch.cuda.synchronize()

    out = torch.empty_like(ref_out)
    scale = torch.empty_like(ref_scale)
    infinicore.deepseek_v4_rms_norm_per_block_quant_(
        infinicore.from_torch(out),
        infinicore.from_torch(x),
        infinicore.from_torch(weight),
        infinicore.from_torch(scale),
        epsilon,
        None,
        None,
        group_size,
        False,
    )
    infinicore.sync_stream()

    assert torch.equal(out, ref_out)
    assert torch.allclose(scale, ref_scale, atol=0, rtol=0)
    print("DeepseekV4RMSNormPerBlockQuant: passed")


if __name__ == "__main__":
    main()
