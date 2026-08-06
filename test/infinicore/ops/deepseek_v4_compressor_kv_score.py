import argparse

import infinicore
import torch
from infinicore.lib import _infinicore


DSV4_HIDDEN = 4096
DSV4_HEAD_DIM = 512
DEFAULT_TOKENS = "1,8,16,32,64,128"


def _parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _wrap(tensor, keepalive):
    wrapped = infinicore.from_torch(tensor)
    keepalive.append(wrapped)
    return wrapped._underlying


def _sync():
    infinicore.sync_stream()
    torch.cuda.synchronize()


def _reference(x, wkv, wgate):
    return torch.cat(
        [
            torch.mm(x, wkv.t()),
            torch.mm(x, wgate.t()),
        ],
        dim=-1,
    )


def _max_rel(got, ref):
    got_f = got.float()
    ref_f = ref.float()
    denom = ref_f.abs().clamp_min(1e-6)
    return ((got_f - ref_f).abs() / denom).max().item()


def _run_case(name, tokens, proj_size, args):
    torch.manual_seed(args.seed + tokens * 17 + proj_size)
    x = torch.randn((tokens, DSV4_HIDDEN), device="cuda", dtype=torch.bfloat16)
    wkv = torch.randn((proj_size, DSV4_HIDDEN), device="cuda", dtype=torch.bfloat16)
    wgate = torch.randn((proj_size, DSV4_HIDDEN), device="cuda", dtype=torch.bfloat16)
    wkv_gate = torch.cat([wkv, wgate], dim=0).contiguous()

    ref = _reference(x, wkv, wgate)
    unpacked = torch.empty_like(ref)
    packed = torch.empty_like(ref)

    keepalive = []
    x_core = _wrap(x, keepalive)
    wkv_core = _wrap(wkv, keepalive)
    wgate_core = _wrap(wgate, keepalive)
    wkv_gate_core = _wrap(wkv_gate, keepalive)
    unpacked_core = _wrap(unpacked, keepalive)
    packed_core = _wrap(packed, keepalive)

    _infinicore.deepseek_v4_compressor_kv_score_unpacked_(unpacked_core, x_core, wkv_core, wgate_core)
    _infinicore.deepseek_v4_compressor_kv_score_packed_(packed_core, x_core, wkv_gate_core)
    _sync()

    unpacked_abs = (unpacked.float() - ref.float()).abs().max().item()
    packed_abs = (packed.float() - ref.float()).abs().max().item()
    packed_vs_unpacked_abs = (packed.float() - unpacked.float()).abs().max().item()
    unpacked_rel = _max_rel(unpacked, ref)
    packed_rel = _max_rel(packed, ref)
    unpacked_ok = torch.allclose(unpacked, ref, atol=args.atol, rtol=args.rtol)
    packed_ok = torch.allclose(packed, ref, atol=args.atol, rtol=args.rtol)
    packed_vs_unpacked_ok = torch.allclose(packed, unpacked, atol=args.atol, rtol=args.rtol)

    print(
        f"{name:>4} tokens={tokens:4d} "
        f"x=[{tokens},{DSV4_HIDDEN}] w=[{proj_size},{DSV4_HIDDEN}] packed_w=[{proj_size * 2},{DSV4_HIDDEN}] "
        f"unpacked_abs={unpacked_abs:.4e} unpacked_rel={unpacked_rel:.4e} unpacked_ok={unpacked_ok} "
        f"packed_abs={packed_abs:.4e} packed_rel={packed_rel:.4e} packed_ok={packed_ok} "
        f"packed_vs_unpacked_abs={packed_vs_unpacked_abs:.4e} packed_vs_unpacked_ok={packed_vs_unpacked_ok}"
    )
    assert unpacked_ok
    assert packed_ok
    assert packed_vs_unpacked_ok


def main():
    parser = argparse.ArgumentParser(description="Validate DeepSeek-V4 compressor kv-score packed/unpacked InfiniCore paths.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--cases", default="c4,c128", help="Comma-separated list from: c4,c128")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()

    case_to_proj = {
        "c4": 2 * DSV4_HEAD_DIM,
        "c128": DSV4_HEAD_DIM,
    }
    tokens_list = _parse_int_list(args.tokens)
    cases = [item.strip().lower() for item in args.cases.split(",") if item.strip()]

    print("DeepSeek-V4 compressor kv-score packed/unpacked correctness")
    print(f"hidden={DSV4_HIDDEN} head_dim={DSV4_HEAD_DIM} tokens={tokens_list} cases={cases}")
    for case in cases:
        if case not in case_to_proj:
            raise ValueError(f"unsupported case: {case}")
        for tokens in tokens_list:
            _run_case(case, tokens, case_to_proj[case], args)
    print("DeepseekV4CompressorKvScore: passed")


if __name__ == "__main__":
    main()
