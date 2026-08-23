import argparse

import infinicore
import torch
import vllm._C  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.parse_args()

    torch.manual_seed(0)
    x = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)

    ref_out = torch.empty_like(x, dtype=torch.int8)
    ref_scale = torch.empty((x.numel() // x.shape[-1], 1), device=x.device, dtype=torch.float32)
    torch.ops._C.dynamic_scaled_int8_quant(ref_out, x, ref_scale, None)
    torch.cuda.synchronize()

    out = torch.empty_like(ref_out)
    scale = torch.empty_like(ref_scale)
    infinicore.deepseek_v4_dynamic_scaled_int8_quant_(
        infinicore.from_torch(out),
        infinicore.from_torch(x),
        infinicore.from_torch(scale),
        None,
    )
    infinicore.sync_stream()

    assert torch.equal(out, ref_out)
    assert torch.allclose(scale, ref_scale, atol=0, rtol=0)
    print("DeepseekV4DynamicScaledInt8Quant: passed")


if __name__ == "__main__":
    main()
