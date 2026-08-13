import argparse
import statistics
import time

import torch

import infinicore
from infinicore.lib import _infinicore as ops


def as_tensor(t: torch.Tensor):
    return infinicore.from_torch(t)._underlying


def random_w8a8_case(tokens: int, hidden: int, out_features: int, device: str):
    torch.manual_seed(20260812)
    x = torch.randn((tokens, hidden), device=device, dtype=torch.bfloat16)
    w = torch.randint(-64, 64, (out_features, hidden), device=device, dtype=torch.int8)
    w_scale = torch.rand((out_features, 1), device=device, dtype=torch.float32) * 0.02 + 0.001
    smooth = torch.ones((hidden,), device=device, dtype=torch.float32)
    out_ref = torch.empty((tokens, out_features), device=device, dtype=torch.bfloat16)
    out_new = torch.empty_like(out_ref)
    q_ref = torch.empty((tokens, hidden), device=device, dtype=torch.int8)
    q_new = torch.empty_like(q_ref)
    scale_ref = torch.empty((tokens, 1), device=device, dtype=torch.float32)
    scale_new = torch.empty_like(scale_ref)
    return x, w, w_scale, smooth, out_ref, out_new, q_ref, q_new, scale_ref, scale_new


def bench(fn, iters: int, warmup: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.mean(times), statistics.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--tokens", type=str, default="17,32,64,128,256")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=1536)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()

    device = "cuda"
    if args.hygon:
        infinicore.set_device(infinicore.device("cuda", 0))
    torch.cuda.set_device(0)

    print("tokens, stable_avg_ms, hipblaslt_avg_ms, speedup, max_abs_diff, allclose")
    for tokens in [int(x) for x in args.tokens.split(",") if x.strip()]:
        x, w, w_scale, smooth, out_ref, out_new, q_ref, q_new, scale_ref, scale_new = random_w8a8_case(
            tokens, args.hidden, args.out_features, device
        )
        weight_t_core = as_tensor(w).permute([1, 0])
        ops.deepseek_v4_lmslim_linear_w8a8_(
            as_tensor(out_ref),
            as_tensor(x),
            weight_t_core,
            as_tensor(w_scale),
            None,
            as_tensor(q_ref),
            as_tensor(scale_ref),
            as_tensor(smooth),
        )
        ops.deepseek_v4_lmslim_hipblaslt_channelwise_linear_w8a8_(
            as_tensor(out_new),
            as_tensor(x),
            as_tensor(w),
            as_tensor(w_scale),
            None,
            as_tensor(q_new),
            as_tensor(scale_new),
            as_tensor(smooth),
        )
        torch.cuda.synchronize()

        diff = (out_ref.float() - out_new.float()).abs().max().item()
        ok = torch.allclose(out_ref, out_new, atol=args.atol, rtol=args.rtol)

        stable_avg, _ = bench(
            lambda: ops.deepseek_v4_lmslim_linear_w8a8_(
                as_tensor(out_ref),
                as_tensor(x),
                weight_t_core,
                as_tensor(w_scale),
                None,
                as_tensor(q_ref),
                as_tensor(scale_ref),
                as_tensor(smooth),
            ),
            args.iters,
            args.warmup,
        )
        hipblaslt_avg, _ = bench(
            lambda: ops.deepseek_v4_lmslim_hipblaslt_channelwise_linear_w8a8_(
                as_tensor(out_new),
                as_tensor(x),
                as_tensor(w),
                as_tensor(w_scale),
                None,
                as_tensor(q_new),
                as_tensor(scale_new),
                as_tensor(smooth),
            ),
            args.iters,
            args.warmup,
        )
        speedup = stable_avg / hipblaslt_avg if hipblaslt_avg > 0 else float("inf")
        print(f"{tokens}, {stable_avg:.6f}, {hipblaslt_avg:.6f}, {speedup:.3f}, {diff:.6f}, {ok}")
        if not ok:
            raise AssertionError(f"tokens={tokens} output mismatch: max_abs_diff={diff}")


if __name__ == "__main__":
    main()
