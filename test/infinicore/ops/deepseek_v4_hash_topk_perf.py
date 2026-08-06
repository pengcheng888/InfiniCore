import argparse
import csv
import time

import infinicore
import torch
from infinicore.lib import _infinicore


DEFAULT_TOKENS = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"


def _parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _wrap(tensor, keepalive):
    wrapped = infinicore.from_torch(tensor)
    keepalive.append(wrapped)
    return wrapped._underlying


def _sync():
    infinicore.sync_stream()


def _bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    _sync()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    total_ms = (time.perf_counter() - start) * 1000.0
    return total_ms, total_ms / float(iters)


def _max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


def _check_against(name, ref_weights, ref_indices, out_weights, out_indices, atol, rtol):
    if not torch.equal(ref_indices, out_indices):
        mismatch = (ref_indices != out_indices).nonzero()
        first = mismatch[0].tolist() if mismatch.numel() else None
        raise AssertionError(f"{name} indices mismatch at {first}")
    if not torch.allclose(ref_weights, out_weights, atol=atol, rtol=rtol):
        raise AssertionError(f"{name} weights mismatch: max_abs={_max_abs(ref_weights, out_weights)}")


def _make_outputs(tokens, topk_fused, device):
    return {
        name: (
            torch.empty(tokens, topk_fused, dtype=torch.float32, device=device),
            torch.empty(tokens, topk_fused, dtype=torch.int32, device=device),
        )
        for name in ("naive", "generic", "sglang", "auto")
    }


def _make_case(tokens, experts, topk, num_fused_shared_experts, vocab_size, tid2eid_dtype, seed):
    torch.manual_seed(seed + tokens * 17 + experts)
    device = "cuda"
    router_logits = torch.randn(tokens, experts, dtype=torch.float32, device=device)
    input_ids = torch.randint(0, vocab_size, (tokens,), dtype=torch.int64, device=device)
    tid2eid = torch.randint(0, experts, (vocab_size, topk), dtype=tid2eid_dtype, device=device)
    outputs = _make_outputs(tokens, topk + num_fused_shared_experts, device)

    keepalive = []
    raw = {
        "router_logits": _wrap(router_logits, keepalive),
        "input_ids": _wrap(input_ids, keepalive),
        "tid2eid": _wrap(tid2eid, keepalive),
    }
    tensors = {}
    for backend, (weights, indices) in outputs.items():
        raw[f"weights_{backend}"] = _wrap(weights, keepalive)
        raw[f"indices_{backend}"] = _wrap(indices, keepalive)
        tensors[backend] = (weights, indices)
    return raw, tensors, keepalive


def _run_case(tokens, args):
    tid2eid_dtype = torch.int32 if args.tid2eid_dtype == "int32" else torch.int64
    raw, tensors, keepalive = _make_case(
        tokens,
        args.experts,
        args.topk,
        args.num_fused_shared_experts,
        args.vocab_size,
        tid2eid_dtype,
        args.seed,
    )

    def run_naive():
        _infinicore.deepseek_v4_hash_topk_naive_(
            raw["weights_naive"],
            raw["indices_naive"],
            raw["router_logits"],
            raw["input_ids"],
            raw["tid2eid"],
            args.num_fused_shared_experts,
            args.routed_scaling_factor,
            args.scoring_func,
        )

    def run_generic():
        _infinicore.deepseek_v4_hash_topk_generic_kernel_(
            raw["weights_generic"],
            raw["indices_generic"],
            raw["router_logits"],
            raw["input_ids"],
            raw["tid2eid"],
            args.num_fused_shared_experts,
            args.routed_scaling_factor,
            args.scoring_func,
        )

    def run_auto():
        _infinicore.deepseek_v4_hash_topk_kernel_(
            raw["weights_auto"],
            raw["indices_auto"],
            raw["router_logits"],
            raw["input_ids"],
            raw["tid2eid"],
            args.num_fused_shared_experts,
            args.routed_scaling_factor,
            args.scoring_func,
        )

    def run_sglang():
        _infinicore.deepseek_v4_hash_topk_sglang_kernel_(
            raw["weights_sglang"],
            raw["indices_sglang"],
            raw["router_logits"],
            raw["input_ids"],
            raw["tid2eid"],
            args.num_fused_shared_experts,
            args.routed_scaling_factor,
            args.scoring_func,
        )

    fns = {
        "naive": run_naive,
        "generic": run_generic,
        "sglang": run_sglang,
        "auto": run_auto,
    }

    if args.check:
        for fn in fns.values():
            fn()
        _sync()
        ref_weights, ref_indices = tensors["naive"]
        for backend in ("generic", "sglang", "auto"):
            out_weights, out_indices = tensors[backend]
            _check_against(backend, ref_weights, ref_indices, out_weights, out_indices, args.atol, args.rtol)

    rows = []
    topk_fused = args.topk + args.num_fused_shared_experts
    for backend, fn in fns.items():
        total_ms, avg_ms = _bench(fn, args.warmup, args.iters)
        rows.append(
            {
                "tokens": tokens,
                "experts": args.experts,
                "topk": args.topk,
                "topk_fused": topk_fused,
                "backend": backend,
                "iters": args.iters,
                "total_ms": total_ms,
                "avg_ms": avg_ms,
            }
        )

    _sync()
    del keepalive
    torch.cuda.empty_cache()
    return rows


def _print_rows(rows):
    indexed = {(row["tokens"], row["backend"]): row for row in rows}
    tokens_list = sorted({row["tokens"] for row in rows})
    header = (
        f"{'tokens':>8}  {'experts':>7}  {'topk':>4}  {'fused':>5}  {'iters':>5}  "
        f"{'naive avg':>10}  {'generic avg':>12}  {'sglang avg':>11}  {'auto avg':>10}  "
        f"{'generic/naive':>13}  {'sglang/naive':>13}  {'auto/naive':>11}"
    )
    print(header)
    print("-" * len(header))
    for tokens in tokens_list:
        naive = indexed.get((tokens, "naive"))
        generic = indexed.get((tokens, "generic"))
        sglang = indexed.get((tokens, "sglang"))
        auto = indexed.get((tokens, "auto"))
        if None in (naive, generic, sglang, auto):
            continue
        speedup_generic = naive["avg_ms"] / generic["avg_ms"] if generic["avg_ms"] > 0 else float("inf")
        speedup_sglang = naive["avg_ms"] / sglang["avg_ms"] if sglang["avg_ms"] > 0 else float("inf")
        speedup_naive = naive["avg_ms"] / auto["avg_ms"] if auto["avg_ms"] > 0 else float("inf")
        print(
            f"{tokens:8d}  {naive['experts']:7d}  {naive['topk']:4d}  {naive['topk_fused']:5d}  {naive['iters']:5d}  "
            f"{naive['avg_ms']:10.6f}  {generic['avg_ms']:12.6f}  {sglang['avg_ms']:11.6f}  {auto['avg_ms']:10.6f}  "
            f"{speedup_generic:12.2f}x  {speedup_sglang:12.2f}x  {speedup_naive:10.2f}x"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--tokens", type=str, default=DEFAULT_TOKENS)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--num-fused-shared-experts", type=int, default=0)
    parser.add_argument("--routed-scaling-factor", type=float, default=1.0)
    parser.add_argument("--scoring-func", type=str, default="sqrtsoftplus")
    parser.add_argument("--vocab-size", type=int, default=129280)
    parser.add_argument("--tid2eid-dtype", choices=("int64", "int32"), default="int64")
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--check", dest="check", action="store_true", default=True)
    parser.add_argument("--no-check", dest="check", action="store_false")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/Hygon device is required for hash_topk perf test")
    if args.experts != 256 or args.topk != 6:
        raise RuntimeError("DSv4 specialized hash_topk perf test expects experts=256 and topk=6")
    if args.scoring_func != "sqrtsoftplus":
        raise RuntimeError("hash_topk currently expects scoring_func=sqrtsoftplus")
    if args.topk + args.num_fused_shared_experts > 32:
        raise RuntimeError("hash_topk perf test expects fused topk <= 32")

    all_rows = []
    for tokens in _parse_int_list(args.tokens):
        all_rows.extend(_run_case(tokens, args))

    _print_rows(all_rows)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["tokens", "experts", "topk", "topk_fused", "backend", "iters", "total_ms", "avg_ms"],
            )
            writer.writeheader()
            writer.writerows(all_rows)


if __name__ == "__main__":
    main()
