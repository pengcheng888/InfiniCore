import argparse
import csv
import time

import infinicore
import torch
from infinicore.lib import _infinicore


DEFAULT_TOKENS = "1,4,16,64,256,1024,4096"


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


def _make_outputs(tokens, topk, device):
    return {
        name: (
            torch.empty(tokens, topk, dtype=torch.float32, device=device),
            torch.empty(tokens, topk, dtype=torch.int32, device=device),
        )
        for name in ("naive", "generic", "auto")
    }


def _make_case(tokens, experts, topk, vocab_size, tid2eid_dtype, seed):
    torch.manual_seed(seed + tokens * 17 + experts)
    device = "cuda"
    logits_hash = torch.randn(tokens, experts, dtype=torch.float32, device=device)
    logits_biased = torch.randn(tokens, experts, dtype=torch.float32, device=device)
    bias = torch.randn(experts, dtype=torch.float32, device=device) * 0.2
    input_ids = torch.randint(0, vocab_size, (tokens,), dtype=torch.int64, device=device)
    tid2eid = torch.randint(0, experts, (vocab_size, topk), dtype=tid2eid_dtype, device=device)

    hash_out = _make_outputs(tokens, topk, device)
    biased_out = _make_outputs(tokens, topk, device)

    keepalive = []
    raw = {
        "logits_hash": _wrap(logits_hash, keepalive),
        "logits_biased": _wrap(logits_biased, keepalive),
        "bias": _wrap(bias, keepalive),
        "input_ids": _wrap(input_ids, keepalive),
        "tid2eid": _wrap(tid2eid, keepalive),
    }
    tensors = {"hash": {}, "biased": {}}
    for backend, (weights, indices) in hash_out.items():
        raw[f"hash_weights_{backend}"] = _wrap(weights, keepalive)
        raw[f"hash_indices_{backend}"] = _wrap(indices, keepalive)
        tensors["hash"][backend] = (weights, indices)
    for backend, (weights, indices) in biased_out.items():
        raw[f"biased_weights_{backend}"] = _wrap(weights, keepalive)
        raw[f"biased_indices_{backend}"] = _wrap(indices, keepalive)
        tensors["biased"][backend] = (weights, indices)
    return raw, tensors, keepalive


def _run_case(tokens, args):
    tid2eid_dtype = torch.int32 if args.tid2eid_dtype == "int32" else torch.int64
    raw, tensors, keepalive = _make_case(
        tokens,
        args.experts,
        args.topk,
        args.vocab_size,
        tid2eid_dtype,
        args.seed,
    )

    def hash_naive():
        _infinicore.deepseek_v4_hash_topk_naive_(
            raw["hash_weights_naive"],
            raw["hash_indices_naive"],
            raw["logits_hash"],
            raw["input_ids"],
            raw["tid2eid"],
            args.renormalize,
        )

    def hash_generic():
        _infinicore.deepseek_v4_hash_topk_generic_kernel_(
            raw["hash_weights_generic"],
            raw["hash_indices_generic"],
            raw["logits_hash"],
            raw["input_ids"],
            raw["tid2eid"],
            args.renormalize,
        )

    def hash_auto():
        _infinicore.deepseek_v4_hash_topk_kernel_(
            raw["hash_weights_auto"],
            raw["hash_indices_auto"],
            raw["logits_hash"],
            raw["input_ids"],
            raw["tid2eid"],
            args.renormalize,
        )

    def biased_naive():
        _infinicore.deepseek_v4_topk_naive_(
            raw["biased_weights_naive"],
            raw["biased_indices_naive"],
            raw["logits_biased"],
            raw["bias"],
            args.renormalize,
        )

    def biased_generic():
        _infinicore.deepseek_v4_topk_generic_kernel_(
            raw["biased_weights_generic"],
            raw["biased_indices_generic"],
            raw["logits_biased"],
            raw["bias"],
            args.renormalize,
        )

    def biased_auto():
        _infinicore.deepseek_v4_topk_kernel_(
            raw["biased_weights_auto"],
            raw["biased_indices_auto"],
            raw["logits_biased"],
            raw["bias"],
            args.renormalize,
        )

    fns = {
        "hash_topk": {
            "naive": hash_naive,
            "generic": hash_generic,
            "auto": hash_auto,
        },
        "biased_topk": {
            "naive": biased_naive,
            "generic": biased_generic,
            "auto": biased_auto,
        },
    }

    if args.check:
        for op_name, op_fns in fns.items():
            op_fns["naive"]()
            op_fns["generic"]()
            op_fns["auto"]()
        _sync()
        for group_name, op_name in (("hash", "hash_topk"), ("biased", "biased_topk")):
            ref_weights, ref_indices = tensors[group_name]["naive"]
            for backend in ("generic", "auto"):
                out_weights, out_indices = tensors[group_name][backend]
                _check_against(f"{op_name}/{backend}", ref_weights, ref_indices, out_weights, out_indices, args.atol, args.rtol)

    rows = []
    for op_name, op_fns in fns.items():
        for backend in ("naive", "generic", "auto"):
            total_ms, avg_ms = _bench(op_fns[backend], args.warmup, args.iters)
            rows.append(
                {
                    "tokens": tokens,
                    "experts": args.experts,
                    "topk": args.topk,
                    "op": op_name,
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


def _print_by_op(rows):
    indexed = {(row["tokens"], row["op"], row["backend"]): row for row in rows}
    ops = []
    for row in rows:
        if row["op"] not in ops:
            ops.append(row["op"])
    tokens_list = sorted({row["tokens"] for row in rows})

    for op in ops:
        print("")
        print(f"op: {op}")
        header = (
            f"{'tokens':>8}  {'experts':>7}  {'topk':>4}  {'iters':>5}  "
            f"{'naive avg':>10}  {'generic avg':>12}  {'auto avg':>10}  "
            f"{'auto/generic':>13}  {'auto/naive':>11}"
        )
        print(header)
        print("-" * len(header))
        for tokens in tokens_list:
            naive = indexed.get((tokens, op, "naive"))
            generic = indexed.get((tokens, op, "generic"))
            auto = indexed.get((tokens, op, "auto"))
            if None in (naive, generic, auto):
                continue
            speedup_generic = generic["avg_ms"] / auto["avg_ms"] if auto["avg_ms"] > 0 else float("inf")
            speedup_naive = naive["avg_ms"] / auto["avg_ms"] if auto["avg_ms"] > 0 else float("inf")
            print(
                f"{tokens:8d}  {naive['experts']:7d}  {naive['topk']:4d}  {naive['iters']:5d}  "
                f"{naive['avg_ms']:10.6f}  {generic['avg_ms']:12.6f}  {auto['avg_ms']:10.6f}  "
                f"{speedup_generic:12.2f}x  {speedup_naive:10.2f}x"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--tokens", type=str, default=DEFAULT_TOKENS)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--tid2eid-dtype", choices=("int64", "int32"), default="int64")
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--renormalize", dest="renormalize", action="store_true", default=True)
    parser.add_argument("--no-renormalize", dest="renormalize", action="store_false")
    parser.add_argument("--check", dest="check", action="store_true", default=True)
    parser.add_argument("--no-check", dest="check", action="store_false")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/Hygon device is required for gate topk perf test")
    if args.experts != 256 or args.topk != 6 or not args.renormalize:
        raise RuntimeError("DSv4 specialized perf test expects experts=256, topk=6, renormalize=true")

    all_rows = []
    for tokens in _parse_int_list(args.tokens):
        all_rows.extend(_run_case(tokens, args))

    _print_by_op(all_rows)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["tokens", "experts", "topk", "op", "backend", "iters", "total_ms", "avg_ms"],
            )
            writer.writeheader()
            writer.writerows(all_rows)


if __name__ == "__main__":
    main()
