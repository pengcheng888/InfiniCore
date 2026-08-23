import ctypes
import glob
import importlib.util
import os
import sys
from typing import Iterable, List


def _candidate_prefixes(path: str) -> List[str]:
    """
    Return HPCC install prefixes to search for libs.
    Prefer HPCC_PATH; if absent and explicitly opted-in, fall back to /opt/hpcc.
    """
    prefixes: List[str] = []
    if path:
        prefixes.append(path)

    seen = set()
    unique: List[str] = []
    for p in prefixes:
        if p and p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _try_load(paths: Iterable[str], name: str) -> bool:
    """Try to load a shared library from given paths or system search path."""
    for path in paths:
        full = os.path.join(path, "lib", name)
        if os.path.exists(full):
            try:
                ctypes.CDLL(full, mode=ctypes.RTLD_GLOBAL)
                return True
            except OSError:
                # Try next candidate
                continue
    # Last resort: rely on loader search path
    try:
        ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        return True
    except OSError:
        return False


def preload_torch() -> None:
    """
    Import torch so its shared libraries are available to ATen-based backends.
    """
    import torch  # noqa: F401 - imported for shared-library loading side effects


def preload_hpcc() -> None:
    """
    Best-effort preload of key HPCC runtime libs with RTLD_GLOBAL.

    This mirrors the behavior of torch's HPCC build that loads libtorch_global_deps.so,
    but avoids introducing a hard torch dependency. All failures are swallowed.
    """
    hpcc_path = os.getenv("HPCC_PATH")
    if not hpcc_path:
        return

    prefixes = _candidate_prefixes(hpcc_path)
    libs = [
        "libhcruntime.so",
        "libhcToolsExt.so",
        "libruntime_cu.so",
        "libhccompiler.so",
    ]

    for lib in libs:
        _try_load(prefixes, lib)


def preload_torch_hip() -> None:
    """
    Best-effort preload of torch HIP runtime libs with RTLD_GLOBAL.

    This helps external extensions resolve c10::hip symbols when they are
    not recorded as direct DT_NEEDED dependencies.
    """
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.origin:
        return
    try:
        __import__("torch")
    except Exception:
        return
    torch_dir = os.path.dirname(spec.origin)
    torch_libdir = os.path.join(torch_dir, "lib")
    if not os.path.isdir(torch_libdir):
        return

    libs = [
        "libtorch_global_deps.so",
        "libc10.so",
        "libc10_hip.so",
        "libtorch_cpu.so",
        "libtorch.so",
        "libtorch_hip.so",
    ]
    for lib in libs:
        full = os.path.join(torch_libdir, lib)
        if os.path.exists(full):
            try:
                ctypes.CDLL(full, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                # Best-effort preload, continue on errors.
                pass


def preload_torch_cuda() -> None:
    """
    Best-effort preload of torch CUDA Python runtime libs with RTLD_GLOBAL.

    ATen-enabled NVIDIA builds can need libtorch_python before torch has been
    imported by the caller. Loading it by absolute path avoids relying on
    transitive RUNPATH resolution during extension import.
    """
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.origin:
        return
    torch_dir = os.path.dirname(spec.origin)
    torch_libdir = os.path.join(torch_dir, "lib")
    if not os.path.isdir(torch_libdir):
        return

    libs = [
        "libtorch_global_deps.so",
        "libc10.so",
        "libc10_cuda.so",
        "libtorch_cpu.so",
        "libtorch_cuda.so",
        "libtorch.so",
        "libtorch_python.so",
    ]
    for lib in libs:
        full = os.path.join(torch_libdir, lib)
        if os.path.exists(full):
            try:
                ctypes.CDLL(full, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                # Best-effort preload, continue on errors.
                pass

    # Importing sgl_kernel registers its torch custom-op schemas and loads the
    # architecture-specific common_ops SO used by Qwen3 bridge operators.
    try:
        __import__("sgl_kernel")
    except Exception:
        pass


_deepseek_v4_lightop_lib = None


def register_deepseek_v4_lightop_ops() -> None:
    """Register stable torch dispatcher wrappers for lightop pybind SO ops."""
    global _deepseek_v4_lightop_lib
    if _deepseek_v4_lightop_lib is not None:
        return
    try:
        import torch
        from lightop import op as lightop_op
    except Exception:
        return

    try:
        lib = torch.library.Library("infinicore_deepseek_v4", "FRAGMENT")
        lib.define(
            "lightop_moe_gemm_marlin_w8a8(Tensor input, Tensor b_qweight, Tensor(a!) output, "
            "Tensor a_scale, Tensor b_scale, Tensor? topk_weights, Tensor sorted_token_ids, "
            "Tensor expert_ids, Tensor num_tokens_post_pad, int top_k, int mode, int delta) -> Tensor(a!)"
        )
        lib.define(
            "lightop_fuse_silu_mul_quant(Tensor input, Tensor(a!) output, Tensor(b!) scales, "
            "Tensor? num_local_tokens_tensor, int topk, int expect_m, Tensor? expert_ids) -> (Tensor(a!), Tensor(b!))"
        )
        lib.define(
            "lightop_moe_sum(Tensor input, Tensor(a!) output, Tensor? bias, Tensor? expert_mask, "
            "Tensor? num_local_tokens, float factor, int expect_m) -> Tensor(a!)"
        )
        lib.define(
            "lightop_moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, "
            "Tensor(a!) sorted_token_ids, Tensor(b!) expert_ids, Tensor(c!) num_tokens_post_pad, "
            "Tensor? expert_map, Tensor? expert_mask, Tensor? num_local_tokens, bool is_ep, "
            "bool is_fuse_fill) -> (Tensor(a!), Tensor(b!), Tensor(c!))"
        )

        def _moe_gemm(input, b_qweight, output, a_scale, b_scale, topk_weights,
                      sorted_token_ids, expert_ids, num_tokens_post_pad, top_k: int,
                      mode: int, delta: int):
            if mode < 1000:
                lightop_op.moe_gemm_marlin_w8a8(
                    input, b_qweight, output, a_scale, b_scale, topk_weights,
                    sorted_token_ids, expert_ids, num_tokens_post_pad, top_k, mode, delta)
            else:
                lightop_op.moe_marlin_w8a8_asm(
                    input, b_qweight, output, a_scale, b_scale, topk_weights,
                    sorted_token_ids, expert_ids, num_tokens_post_pad, top_k, mode, delta)
            return output

        def _fuse_silu_mul_quant(input, output, scales, num_local_tokens_tensor=None,
                                 topk: int = 1, expect_m: int = -1, expert_ids=None):
            lightop_op.fuse_silu_mul_quant(
                input, output, scales, num_local_tokens_tensor, topk, expect_m, expert_ids)
            return output, scales

        def _moe_sum(input, output, bias=None, expert_mask=None, num_local_tokens=None,
                     factor: float = 1.0, expect_m: int = -1):
            lightop_op.moe_sum(input, output, bias, expert_mask, num_local_tokens, factor, expect_m)
            return output

        def _moe_align(topk_ids, num_experts: int, block_size: int, sorted_token_ids,
                       expert_ids, num_tokens_post_pad, expert_map=None, expert_mask=None,
                       num_local_tokens=None, is_ep: bool = False, is_fuse_fill: bool = True):
            lightop_op.moe_align_block_size(
                topk_ids,
                num_experts,
                block_size,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_pad,
                expert_map=expert_map,
                expert_mask=expert_mask,
                num_local_tokens=num_local_tokens,
                Is_EP=is_ep,
                Is_fuse_fill=is_fuse_fill,
            )
            return sorted_token_ids, expert_ids, num_tokens_post_pad

        lib.impl("lightop_moe_gemm_marlin_w8a8", _moe_gemm, "CUDA")
        lib.impl("lightop_fuse_silu_mul_quant", _fuse_silu_mul_quant, "CUDA")
        lib.impl("lightop_moe_sum", _moe_sum, "CUDA")
        lib.impl("lightop_moe_align_block_size", _moe_align, "CUDA")
        _deepseek_v4_lightop_lib = lib
    except Exception:
        _deepseek_v4_lightop_lib = None

def _prefer_rocm_vllm_platform() -> None:
    """Avoid vLLM cuda/rocm double-plugin activation in mixed Hygon envs."""
    try:
        import vllm.platforms as platforms

        platforms.builtin_platform_plugins["cuda"] = lambda: None
    except Exception:
        pass


def preload_deepseek_v4_extensions() -> None:
    """
    Best-effort import of Hygon DeepSeek-V4 extension modules.

    Importing these modules registers their torch custom-op schemas. InfiniCore
    C++ bridge operators resolve the schemas through c10::Dispatcher at call
    time, so InfiniLM must preload them before running C++ forward. Import
    vllm._C directly; vllm._custom_ops can activate conflicting platform
    plugins in this mixed CUDA/ROCm environment.
    """
    _prefer_rocm_vllm_platform()
    for module_name in (
        "sgl_kernel",
        "aiter",
        "vllm._C",
        "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors_moe_marlin",
    ):
        try:
            __import__(module_name)
        except Exception:
            pass
    register_deepseek_v4_lightop_ops()


def preload_flash_attn() -> None:
    """
    Best-effort preload of flash_attn_2_cuda extension with RTLD_GLOBAL.

    InfiniCore hygon wrapper resolves C symbols like `mha_varlen_fwd` from the
    flash-attn extension at runtime via dlsym(RTLD_DEFAULT, ...). The symbols
    only need to be available when the operator is actually called, not at
    library load time. So this preload is a convenience — if it fails, the
    symbols will be resolved later when torch + flash_attn are imported by
    the application (e.g. InfiniLM).
    """
    candidates: List[str] = []
    from_env = os.getenv("FLASH_ATTN_PREBUILT")
    if from_env:
        if os.path.isfile(from_env):
            candidates.append(from_env)
        elif os.path.isdir(from_env):
            candidates.extend(
                glob.glob(os.path.join(from_env, "flash_attn_2_cuda*.so"))
            )

    # Try resolving via Python import metadata.
    spec = importlib.util.find_spec("flash_attn_2_cuda")
    if spec and spec.origin and os.path.exists(spec.origin):
        candidates.append(spec.origin)

    # Fallback: scan python paths for extension module.
    for p in sys.path:
        if not p:
            continue
        candidates.extend(glob.glob(os.path.join(p, "flash_attn_2_cuda*.so")))

    # Common installation locations.
    candidates.extend(
        glob.glob("/usr/local/lib/python*/dist-packages/flash_attn_2_cuda*.so")
    )
    candidates.extend(glob.glob("/root/.infini/lib/flash_attn_2_cuda*.so"))

    seen = set()
    for so_path in candidates:
        if not so_path or so_path in seen:
            continue
        seen.add(so_path)
        if not os.path.exists(so_path):
            continue
        try:
            ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


def preload_cambricon() -> None:
    """Import torch and torch_mlu for shared-library and backend setup."""
    preload_torch()
    try:
        import torch_mlu  # noqa: F401
    except Exception:
        # The linked rpaths still allow the C++ library to load. Backend
        # registration remains best-effort at package import time.
        pass


def _should_preload_device(device_type: str) -> bool:
    """
    Check if preload is needed for a specific device type.
    """
    device_env_map = {
        "METAX": ["MACA_PATH", "HPCC_PATH", "INFINICORE_PRELOAD_HPCC"],
        "HYGON": ["DTK_ROOT", "INFINICORE_PRELOAD_TORCH_HIP"],
        "ASCEND": ["ASCEND_HOME", "ASCEND_TOOLKIT_HOME"],
        "CAMBRICON": ["NEUWARE_HOME", "INFINICORE_PRELOAD_CAMBRICON"],
        "NVIDIA": ["CUDA_HOME", "CUDA_PATH", "INFINICORE_PRELOAD_TORCH_CUDA"],
        # Add other device types here as needed:
    }

    env_vars = device_env_map.get(device_type, [])
    for env_var in env_vars:
        if os.getenv(env_var):
            return True
    if device_type == "HYGON":
        dtk_root = os.getenv("DTK_ROOT") or "/opt/dtk"
        if os.path.isdir(dtk_root):
            return True
    if device_type == "CAMBRICON":
        return importlib.util.find_spec("torch_mlu") is not None
    return False


def preload_device(device_type: str) -> None:
    """
    Preload runtime libraries for a specific device type if needed.

    Args:
        device_type: Device type name (e.g., "METAX", "ASCEND", etc.)
    """
    if device_type == "METAX":
        preload_hpcc()
        preload_torch()
    elif device_type == "HYGON":
        preload_torch()
        preload_torch_hip()
        preload_deepseek_v4_extensions()
        preload_flash_attn()
    elif device_type == "ASCEND":
        preload_torch()
    elif device_type == "NVIDIA":
        preload_torch_cuda()
    # Add other device preload functions here as needed:
    elif device_type == "CAMBRICON":
        preload_cambricon()
    # etc.


def preload() -> None:
    """
    Universal preload function that loops through device types and preloads when required.

    This function detects available device types and preloads their runtime libraries
    if the environment indicates they are needed.
    """
    # Device types that may require preload. Keep Hygon-only preloads gated by
    # Hygon environment markers so other CUDA-compatible platforms do not load
    # unrelated torch/flash-attn libraries during package import.
    device_types = [
        "METAX",  # HPCC/METAX
        "HYGON",
        "ASCEND",
        "CAMBRICON",
        "NVIDIA",
        # Add other device types here as they are implemented:
        # etc.
    ]

    for device_type in device_types:
        if _should_preload_device(device_type):
            try:
                preload_device(device_type)
            except Exception:
                # Swallow all errors - preload is best-effort
                pass
