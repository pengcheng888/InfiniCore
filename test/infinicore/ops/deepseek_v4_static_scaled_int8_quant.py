import argparse

import infinicore
import torch
import vllm._C  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.parse_args()

    torch.manual_seed(1)
    x = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor(0.05, device="cuda", dtype=torch.float32)

    ref_out = torch.empty_like(x, dtype=torch.int8)
    torch.ops._C.static_scaled_int8_quant(ref_out, x, scale, None)
    torch.cuda.synchronize()

    out = torch.empty_like(ref_out)
    infinicore.deepseek_v4_static_scaled_int8_quant_(
        infinicore.from_torch(out),
        infinicore.from_torch(x),
        infinicore.from_torch(scale),
        None,
    )
    infinicore.sync_stream()

    assert torch.equal(out, ref_out)
    print("DeepseekV4StaticScaledInt8Quant: passed")


if __name__ == "__main__":
    main()
