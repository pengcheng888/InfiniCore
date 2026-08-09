import argparse
import json
import statistics
import time

import infinicore
import torch
from infinicore.lib import _infinicore


HEAD_DIM = 128
ROPE_DIM = 64
VALUE_BYTES_PER_TOKEN = 128
SCALE_BYTES_PER_TOKEN = 4
DEFAULT_MODEL_CONFIG = "/data/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8/config.json"
DEFAULT_TOKENS = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"
DEFAULT_PAGE_SIZE = 64
DEFAULT_NUM_BLOCKS = 512
DEFAULT_BLOCK_SIZE = 256
DEFAULT_PROMPT_TOKENS = 7
DEFAULT_DECODE_STEPS = 15
DEFAULT_COMPRESS_RATIO = 4
DEFAULT_MAX_POS = 1048576


def _parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _page_bytes(page_size):
    return (VALUE_BYTES_PER_TOKEN + SCALE_BYTES_PER_TOKEN) * page_size


def _load_model_defaults(path):
    defaults = {
        "head_dim": HEAD_DIM,
        "rope_dim": ROPE_DIM,
        "max_pos": DEFAULT_MAX_POS,
        "eps": 1e-6,
    }
    if not path:
        return defaults
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if "text_config" in config:
        config = config["text_config"]
    defaults["head_dim"] = int(config.get("index_head_dim", defaults["head_dim"]))
    defaults["rope_dim"] = int(config.get("qk_rope_head_dim", defaults["rope_dim"]))
    defaults["max_pos"] = int(config.get("max_position_embeddings", defaults["max_pos"]))
    defaults["eps"] = float(config.get("rms_norm_eps", defaults["eps"]))
    return defaults


def _as_core(tensor, keepalive):
    base = infinicore.from_torch(tensor)
    wrapped = base.as_strided(list(tensor.shape), list(tensor.stride()))
    keepalive.append(base)
    keepalive.append(wrapped)
    return wrapped._underlying


def _sync():
    infinicore.sync_stream()
    torch.cuda.synchronize()


def _bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    _sync()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.mean(samples), statistics.median(samples)


def _make_freqs(max_pos, device):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, ROPE_DIM, 2, device=device, dtype=torch.float32) / ROPE_DIM))
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    angles = torch.outer(t, inv_freq)
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1).flatten(-2).contiguous()


def _full_slot_for_position(position, block_size):
    block_id = position // block_size
    block_offset = position % block_size
    return block_id * block_size + block_offset


def _make_c4_metadata(raw_positions, args, device):
    clipped_positions = []
    out_locs = []
    for position in raw_positions:
        clipped_positions.append((position // args.compress_ratio) * args.compress_ratio)
        if (position + 1) % args.compress_ratio == 0:
            slot = _full_slot_for_position(position, args.block_size)
            out_locs.append(slot // args.compress_ratio)
        else:
            out_locs.append(-1)
    positions = torch.tensor(clipped_positions, device=device, dtype=torch.int32).contiguous()
    out_loc = torch.tensor(out_locs, device=device, dtype=torch.int32).contiguous()
    valid_out_loc = sum(1 for loc in out_locs if loc >= 0)
    return positions, out_loc, valid_out_loc


def _make_case(tokens, freqs, args, start_position=0, case_name=None):
    torch.manual_seed(args.seed + tokens * 17 + start_position)
    device = "cuda"
    kv = (torch.randn(tokens, HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.2).contiguous()
    weight = (torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.25).contiguous()
    raw_positions = list(range(start_position, start_position + tokens))
    positions, out_loc, valid_out_loc = _make_c4_metadata(raw_positions, args, device)
    max_loc = max((loc for loc in out_loc.cpu().tolist() if loc >= 0), default=-1)
    blocks_needed = max(1, (max_loc + args.page_size) // args.page_size)
    blocks = max(args.num_blocks, blocks_needed)
    baseline_kv = kv.clone()
    baseline_cache = torch.zeros((blocks, _page_bytes(args.page_size)), device=device, dtype=torch.uint8)
    fused_cache = torch.zeros_like(baseline_cache)

    keepalive = []
    core = {
        "baseline_kv": _as_core(baseline_kv, keepalive),
        "kv": _as_core(kv, keepalive),
        "weight": _as_core(weight, keepalive),
        "freqs": _as_core(freqs, keepalive),
        "positions": _as_core(positions, keepalive),
        "out_loc": _as_core(out_loc, keepalive),
        "baseline_cache": _as_core(baseline_cache, keepalive),
        "fused_cache": _as_core(fused_cache, keepalive),
    }
    tensors = {
        "kv": kv,
        "baseline_kv": baseline_kv,
        "baseline_cache": baseline_cache,
        "fused_cache": fused_cache,
        "valid_out_loc": valid_out_loc,
        "case": case_name or str(tokens),
    }
    return tensors, core, keepalive


def _run_baseline(core, eps, page_size):
    _infinicore.deepseek_v4_compress_fused_norm_rope_(
        core["baseline_kv"],
        core["weight"],
        eps,
        core["freqs"],
        core["positions"],
    )
    _infinicore.deepseek_v4_indexer_rotate_(
        core["baseline_kv"],
        True,
    )
    _infinicore.deepseek_v4_store_indexer_raw_cache_(
        core["baseline_kv"],
        core["baseline_cache"],
        core["out_loc"],
        page_size,
    )


def _run_fused(core, eps, page_size):
    _infinicore.deepseek_v4_indexer_compress_norm_rope_store_(
        core["kv"],
        core["weight"],
        eps,
        core["freqs"],
        core["positions"],
        core["out_loc"],
        core["fused_cache"],
        page_size,
    )


def _run_case(case, freqs, args):
    tensors, core, keepalive = _make_case(
        case["tokens"],
        freqs,
        args,
        start_position=case["start_position"],
        case_name=case["name"],
    )
    ok = "skip"
    diff = -1
    if args.check:
        _run_baseline(core, args.eps, args.page_size)
        _run_fused(core, args.eps, args.page_size)
        _sync()
        diff = (tensors["baseline_cache"] != tensors["fused_cache"]).sum().item()
        if args.require_legacy_exact and diff != 0:
            raise AssertionError(f"cache byte mismatch: case={case['name']}, tokens={case['tokens']}, diff_bytes={diff}")
        ok = "True" if diff == 0 else "legacy-diff"
        tensors["baseline_kv"].copy_(tensors["kv"])
        tensors["baseline_cache"].zero_()
        tensors["fused_cache"].zero_()
        _sync()

    def baseline_fn():
        _run_baseline(core, args.eps, args.page_size)

    def fused_fn():
        _run_fused(core, args.eps, args.page_size)

    baseline_avg, baseline_med = _bench(baseline_fn, args.warmup, args.iters)
    fused_avg, fused_med = _bench(fused_fn, args.warmup, args.iters)
    del keepalive
    return {
        "case": tensors["case"],
        "tokens": case["tokens"],
        "valid_out_loc": tensors["valid_out_loc"],
        "baseline_avg": baseline_avg,
        "baseline_med": baseline_med,
        "fused_avg": fused_avg,
        "fused_med": fused_med,
        "avg_speedup": baseline_avg / fused_avg if fused_avg > 0 else float("inf"),
        "med_speedup": baseline_med / fused_med if fused_med > 0 else float("inf"),
        "diff": diff,
        "ok": ok,
    }


def _make_cases(args):
    if args.workload == "infer-default":
        cases = [{"name": "prefill", "tokens": args.prompt_tokens, "start_position": 0}]
        for step in range(args.decode_steps):
            cases.append(
                {
                    "name": f"decode{step + 1}",
                    "tokens": 1,
                    "start_position": args.prompt_tokens + step,
                }
            )
        return cases
    return [
        {"name": f"sweep{tokens}", "tokens": tokens, "start_position": args.start_position}
        for tokens in _parse_int_list(args.tokens)
    ]


def main():
    parser = argparse.ArgumentParser(description="Benchmark DeepSeek-V4 indexer compress_norm_rope_store.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--workload", choices=("infer-default", "sweep"), default="sweep")
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--num-blocks", type=int, default=DEFAULT_NUM_BLOCKS)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--decode-steps", type=int, default=DEFAULT_DECODE_STEPS)
    parser.add_argument("--compress-ratio", type=int, default=DEFAULT_COMPRESS_RATIO)
    parser.add_argument("--start-position", type=int, default=0)
    parser.add_argument("--max-pos", type=int)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--check", action="store_true", default=True)
    parser.add_argument("--no-check", dest="check", action="store_false")
    parser.add_argument("--require-legacy-exact", action="store_true")
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()

    model_defaults = _load_model_defaults(args.model_config)
    if model_defaults["head_dim"] != HEAD_DIM or model_defaults["rope_dim"] != ROPE_DIM:
        raise ValueError(
            f"this benchmark is specialized for index_head_dim={HEAD_DIM}, qk_rope_head_dim={ROPE_DIM}; "
            f"got index_head_dim={model_defaults['head_dim']}, qk_rope_head_dim={model_defaults['rope_dim']}"
        )
    if args.max_pos is None:
        args.max_pos = model_defaults["max_pos"]
    if args.eps is None:
        args.eps = model_defaults["eps"]
    if args.compress_ratio != DEFAULT_COMPRESS_RATIO:
        raise ValueError("deepseek_v4_indexer_compress_norm_rope_store_perf only supports C4 compress_ratio=4")

    cases = _make_cases(args)
    freqs = _make_freqs(args.max_pos, "cuda")
    print("DeepSeek-V4 indexer compress_norm_rope_store 性能测试")
    print(
        f"workload={args.workload} model_config={args.model_config} head_dim={HEAD_DIM} rope_dim={ROPE_DIM} "
        f"page_size={args.page_size} num_blocks={args.num_blocks} block_size={args.block_size} "
        f"compress_ratio={args.compress_ratio} max_pos={args.max_pos} eps={args.eps} "
        f"iters={args.iters} warmup={args.warmup} check={args.check}"
    )
    print(
        f"{'case':>10} | {'tokens':>8} | {'valid':>5} | {'baseline avg':>12} | {'fused avg':>10} | {'avg spd':>7} | "
        f"{'baseline med':>12} | {'fused med':>10} | {'med spd':>7} | {'diff':>6} | {'ok':>5}"
    )
    print("-" * 121)
    for case in cases:
        result = _run_case(case, freqs, args)
        print(
            f"{result['case']:>10} | {result['tokens']:8d} | {result['valid_out_loc']:5d} | "
            f"{result['baseline_avg']:12.4f} | {result['fused_avg']:10.4f} | "
            f"{result['avg_speedup']:7.2f} | {result['baseline_med']:12.4f} | {result['fused_med']:10.4f} | "
            f"{result['med_speedup']:7.2f} | {result['diff']:6d} | {result['ok']:>5}"
        )


if __name__ == "__main__":
    main()
