import argparse
import time

import infinicore
import torch
import torch.nn.functional as F
from infinicore.lib import _infinicore


DEFAULT_TOKENS = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096"


def _parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _wrap(tensor, keepalive):
    wrapped = infinicore.from_torch(tensor)
    keepalive.append(wrapped)
    return wrapped._underlying


def _sync():
    infinicore.sync_stream()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    _sync()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - start) * 1000.0 / float(iters)


def _make_case(tokens, args):
    torch.manual_seed(args.seed + tokens * 17 + args.vocab_size + args.hidden)
    device = "cuda"
    input_ids = torch.randint(0, args.vocab_size, (tokens,), device=device, dtype=torch.int64)
    weight = torch.randn((args.vocab_size, args.hidden), device=device, dtype=torch.bfloat16)
    output_shape = (tokens, args.hc_mult, args.hidden)
    generic_tmp = torch.empty((tokens, args.hidden), device=device, dtype=weight.dtype)
    generic_out = torch.empty(output_shape, device=device, dtype=weight.dtype)
    deepseek_out = torch.empty_like(generic_out)
    kernel_out = torch.empty_like(generic_out)
    naive_out = torch.empty_like(generic_out)

    keepalive = []
    raw = {
        "input": _wrap(input_ids, keepalive),
        "weight": _wrap(weight, keepalive),
        "generic_tmp": _wrap(generic_tmp, keepalive),
        "generic_out": _wrap(generic_out, keepalive),
        "deepseek_out": _wrap(deepseek_out, keepalive),
        "kernel_out": _wrap(kernel_out, keepalive),
        "naive_out": _wrap(naive_out, keepalive),
    }
    return input_ids, weight, generic_tmp, generic_out, deepseek_out, kernel_out, naive_out, raw, keepalive


def _run_case(tokens, args):
    input_ids, weight, generic_tmp, generic_out, deepseek_out, kernel_out, naive_out, raw, keepalive = _make_case(tokens, args)

    def torch_embedding_and_hc_expand():
        embedded = F.embedding(input_ids, weight)
        return embedded.unsqueeze(1).expand(-1, args.hc_mult, -1).contiguous()

    def generic_embedding():
        _infinicore.embedding_(raw["generic_tmp"], raw["input"], raw["weight"])
        generic_out.copy_(generic_tmp.unsqueeze(1).expand(-1, args.hc_mult, -1))

    def deepseek_embedding():
        _infinicore.deepseek_v4_embedding_and_hc_expand_(raw["deepseek_out"], raw["input"], raw["weight"], args.hc_mult)

    def kernel_embedding():
        _infinicore.deepseek_v4_embedding_and_hc_expand_kernel_(raw["kernel_out"], raw["input"], raw["weight"], args.hc_mult)

    def naive_embedding():
        _infinicore.deepseek_v4_embedding_and_hc_expand_naive_(raw["naive_out"], raw["input"], raw["weight"], args.hc_mult)

    max_abs_generic = float("nan")
    max_abs_deepseek = float("nan")
    max_abs_kernel = float("nan")
    max_abs_naive = float("nan")
    generic_ok = "skip"
    deepseek_ok = "skip"
    kernel_ok = "skip"
    naive_ok = "skip"
    if args.check:
        ref = torch_embedding_and_hc_expand()
        generic_embedding()
        deepseek_embedding()
        kernel_embedding()
        naive_embedding()
        _sync()
        max_abs_generic = (ref.float() - generic_out.float()).abs().max().item()
        max_abs_deepseek = (ref.float() - deepseek_out.float()).abs().max().item()
        max_abs_kernel = (ref.float() - kernel_out.float()).abs().max().item()
        max_abs_naive = (ref.float() - naive_out.float()).abs().max().item()
        generic_ok = str(torch.equal(ref, generic_out))
        deepseek_ok = str(torch.equal(ref, deepseek_out))
        kernel_ok = str(torch.equal(ref, kernel_out))
        naive_ok = str(torch.equal(ref, naive_out))

    torch_embedding_ms = _bench(torch_embedding_and_hc_expand, args.warmup, args.iters)
    generic_ms = _bench(generic_embedding, args.warmup, args.iters)
    deepseek_ms = _bench(deepseek_embedding, args.warmup, args.iters)
    kernel_ms = _bench(kernel_embedding, args.warmup, args.iters)
    naive_ms = _bench(naive_embedding, args.warmup, args.iters)

    del keepalive
    torch.cuda.empty_cache()
    return {
        "tokens": tokens,
        "torch_embedding_ms": torch_embedding_ms,
        "generic_ms": generic_ms,
        "deepseek_ms": deepseek_ms,
        "kernel_ms": kernel_ms,
        "naive_ms": naive_ms,
        "generic_speedup": torch_embedding_ms / generic_ms if generic_ms > 0 else float("inf"),
        "deepseek_speedup": torch_embedding_ms / deepseek_ms if deepseek_ms > 0 else float("inf"),
        "kernel_speedup": torch_embedding_ms / kernel_ms if kernel_ms > 0 else float("inf"),
        "naive_speedup": torch_embedding_ms / naive_ms if naive_ms > 0 else float("inf"),
        "max_abs_generic": max_abs_generic,
        "max_abs_deepseek": max_abs_deepseek,
        "max_abs_kernel": max_abs_kernel,
        "max_abs_naive": max_abs_naive,
        "generic_ok": generic_ok,
        "deepseek_ok": deepseek_ok,
        "kernel_ok": kernel_ok,
        "naive_ok": naive_ok,
    }


def _fmt_float(value, digits=4):
    return "nan" if value != value else f"{value:.{digits}f}"


def _fmt_sci(value):
    return "nan" if value != value else f"{value:.4e}"


def main():
    parser = argparse.ArgumentParser(description="Benchmark DeepSeek-V4 embedding lookup.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--vocab-size", type=int, default=129280)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--hc-mult", type=int, default=4)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--check", action="store_true", default=True)
    parser.add_argument("--no-check", dest="check", action="store_false")
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("deepseek_v4_embedding_and_hc_expand_perf requires an accelerator device.")

    print("DeepSeek-V4 embedding + HC expand performance")
    print(
        f"vocab_size={args.vocab_size} hidden={args.hidden} hc_mult={args.hc_mult} "
        f"iters={args.iters} warmup={args.warmup} check={args.check}"
    )
    print(
        f"{'tokens':>8} | {'torch base':>10} | "
        f"{'generic':>10} | {'generic spd':>11} | {'deepseek':>10} | "
        f"{'deepseek spd':>12} | {'kernel':>10} | {'kernel spd':>10} | "
        f"{'naive':>10} | {'naive spd':>10} | {'gen_abs':>10} | "
        f"{'gen_ok':>6} | {'dsv4_abs':>10} | {'dsv4_ok':>7} | "
        f"{'ker_abs':>10} | {'ker_ok':>6} | {'naive_abs':>10} | {'naive_ok':>8}"
    )
    print("-" * 189)
    for tokens in _parse_int_list(args.tokens):
        result = _run_case(tokens, args)
        print(
            f"{result['tokens']:8d} | "
            f"{_fmt_float(result['torch_embedding_ms']):>10} | "
            f"{_fmt_float(result['generic_ms']):>10} | "
            f"{_fmt_float(result['generic_speedup'], 2):>10}x | "
            f"{_fmt_float(result['deepseek_ms']):>10} | "
            f"{_fmt_float(result['deepseek_speedup'], 2):>11}x | "
            f"{_fmt_float(result['kernel_ms']):>10} | "
            f"{_fmt_float(result['kernel_speedup'], 2):>9}x | "
            f"{_fmt_float(result['naive_ms']):>10} | "
            f"{_fmt_float(result['naive_speedup'], 2):>9}x | "
            f"{_fmt_sci(result['max_abs_generic']):>10} | "
            f"{result['generic_ok']:>6} | "
            f"{_fmt_sci(result['max_abs_deepseek']):>10} | "
            f"{result['deepseek_ok']:>7} | "
            f"{_fmt_sci(result['max_abs_kernel']):>10} | "
            f"{result['kernel_ok']:>6} | "
            f"{_fmt_sci(result['max_abs_naive']):>10} | "
            f"{result['naive_ok']:>8}"
        )


if __name__ == "__main__":
    main()
