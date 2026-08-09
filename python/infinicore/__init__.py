import contextlib

with contextlib.suppress(ImportError):
    from ._preload import preload

    preload()

import infinicore.context as context
import infinicore.nn as nn
from infinicore._tensor_str import printoptions, set_printoptions

# Import context functions
from infinicore.context import (
    get_device,
    get_device_count,
    get_stream,
    is_graph_recording,
    set_device,
    start_graph_recording,
    stop_graph_recording,
    sync_device,
    sync_stream,
)
from infinicore.device import device
from infinicore.device_event import DeviceEvent
from infinicore.dtype import (
    bfloat16,
    bool,
    cdouble,
    cfloat,
    chalf,
    complex32,
    complex64,
    complex128,
    double,
    dtype,
    float,
    float16,
    float32,
    float64,
    half,
    int,
    int8,
    int16,
    int32,
    int64,
    long,
    short,
    uint8,
)
from infinicore.ops.acos import acos
from infinicore.ops.add import add
from infinicore.ops.add_rms_norm import add_rms_norm
from infinicore.ops.addbmm import addbmm
from infinicore.ops.addcmul import addcmul
from infinicore.ops.addr import addr
from infinicore.ops.all import all
from infinicore.ops.argwhere import argwhere
from infinicore.ops.asin import asin
from infinicore.ops.asinh import asinh
from infinicore.ops.asum import asum
from infinicore.ops.atanh import atanh
from infinicore.ops.attention import attention
from infinicore.ops.axpy import axpy
from infinicore.ops.baddbmm import baddbmm
from infinicore.ops.bilinear import bilinear
from infinicore.ops.binary_cross_entropy_with_logits import (
    binary_cross_entropy_with_logits,
)
from infinicore.ops.bitwise_right_shift import bitwise_right_shift
from infinicore.ops.blas_amax import blas_amax
from infinicore.ops.blas_amin import blas_amin
from infinicore.ops.blas_copy import blas_copy
from infinicore.ops.blas_dot import blas_dot
from infinicore.ops.block_diag import block_diag
from infinicore.ops.broadcast_to import broadcast_to
from infinicore.ops.cat import cat
from infinicore.ops.cdist import cdist
from infinicore.ops.conv2d import conv2d
from infinicore.ops.cross_entropy import cross_entropy
from infinicore.ops.diff import diff
from infinicore.ops.digamma import digamma
from infinicore.ops.dist import dist
from infinicore.ops.equal import equal
from infinicore.ops.flipud import flipud
from infinicore.ops.float_power import float_power
from infinicore.ops.floor import floor
from infinicore.ops.floor_divide import floor_divide
from infinicore.ops.fmin import fmin
from infinicore.ops.fmod import fmod
from infinicore.ops.hypot import hypot
from infinicore.ops.index_add import index_add
from infinicore.ops.index_copy import index_copy
from infinicore.ops.inner import inner
from infinicore.ops.kron import kron
from infinicore.ops.kthvalue import kthvalue
from infinicore.ops.kv_caching import kv_caching
from infinicore.ops.ldexp import ldexp
from infinicore.ops.lerp import lerp
from infinicore.ops.logaddexp import logaddexp
from infinicore.ops.logaddexp2 import logaddexp2
from infinicore.ops.logcumsumexp import logcumsumexp
from infinicore.ops.logdet import logdet
from infinicore.ops.logical_and import logical_and
from infinicore.ops.logical_not import logical_not
from infinicore.ops.masked_select import masked_select
from infinicore.ops.matmul import matmul
from infinicore.ops.mha import mha
from infinicore.ops.mha_kvcache import mha_kvcache
from infinicore.ops.mha_varlen import mha_varlen
from infinicore.ops.moore_mate_flash_attn import (
    moore_mate_flash_attn_decode,
    moore_mate_flash_attn_prefill,
)
from infinicore.ops.mrope import mrope
from infinicore.ops.mul import mul
from infinicore.ops.mul_scalar import mul_scalar
from infinicore.ops.narrow import narrow
from infinicore.ops.nrm2 import nrm2
from infinicore.ops.paged_attention import paged_attention
from infinicore.ops.paged_attention_prefill import paged_attention_prefill
from infinicore.ops.paged_caching import paged_caching
from infinicore.ops.qwen3_add_rms_norm import qwen3_add_rms_norm, qwen3_add_rms_norm_inplace
from infinicore.ops.qwen3_fused_qk_norm_rope import qwen3_fused_qk_norm_rope, qwen3_fused_qk_norm_rope_
from infinicore.ops.qwen3_mha_kvcache import qwen3_mha_kvcache
from infinicore.ops.qwen3_mha_varlen import qwen3_mha_varlen
from infinicore.ops.qwen3_rms_norm import qwen3_rms_norm
from infinicore.ops.qwen3_rotary_embedding import qwen3_rotary_embedding, qwen3_rotary_embedding_
from infinicore.ops.qwen3_silu_and_mul import qwen3_silu_and_mul
from infinicore.ops.qwen3_store_kvcache import qwen3_store_kvcache, qwen3_store_kvcache_
from infinicore.ops.deepseek_v4_add_rms_norm import deepseek_v4_add_rms_norm, deepseek_v4_add_rms_norm_inplace
from infinicore.ops.deepseek_v4_assign_extend_cache_locs import deepseek_v4_assign_extend_cache_locs_
from infinicore.ops.deepseek_v4_assign_req_to_token_pool import deepseek_v4_assign_req_to_token_pool_
from infinicore.ops.deepseek_v4_concat_and_cache_mla import deepseek_v4_concat_and_cache_mla_
from infinicore.ops.deepseek_v4_create_chunked_prefix_cache_kv_indices import deepseek_v4_create_chunked_prefix_cache_kv_indices_
from infinicore.ops.deepseek_v4_create_flashmla_kv_indices import deepseek_v4_create_flashmla_kv_indices_
from infinicore.ops.deepseek_v4_dcu_cache_alloc import deepseek_v4_dcu_alloc_decode_kernel_, deepseek_v4_dcu_alloc_extend_kernel_
from infinicore.ops.deepseek_v4_deep_gemm import deepseek_v4_deep_gemm_low_latency_grouped_gemm_, deepseek_v4_deep_gemm_moe_w8a8_i8_marlin_prefill_down_, deepseek_v4_deep_gemm_moe_w8a8_marlin_decode_down_fp8_
from infinicore.ops.deepseek_v4_dynamic_scaled_int8_quant import deepseek_v4_dynamic_scaled_int8_quant_
from infinicore.ops.deepseek_v4_fast_topk import deepseek_v4_fast_topk_, deepseek_v4_fast_topk_transform_fused_, deepseek_v4_fast_topk_transform_ragged_fused_
from infinicore.ops.deepseek_v4_flashmla_cache import deepseek_v4_flashmla_cache_indexer_, deepseek_v4_fused_store_flashmla_cache_, deepseek_v4_indexer_rotate_, deepseek_v4_store_flashmla_raw_cache_, deepseek_v4_store_indexer_raw_cache_
from infinicore.ops.deepseek_v4_fused_experts_impl_int8_marlin import deepseek_v4_fused_experts_impl_int8_marlin_, deepseek_v4_python_fused_experts_impl_int8_marlin_
from infinicore.ops.deepseek_v4_flashmla_compute import (
    deepseek_v4_c128_compress_stateful,
    deepseek_v4_c4_compress_stateful,
    deepseek_v4_compress_fused_norm_rope_,
    deepseek_v4_flashmla_sparse_attention_,
    deepseek_v4_flashmla_sparse_attention_metadata_,
    deepseek_v4_flashmla_sparse_attention_out_workspace_,
    deepseek_v4_flashmla_sparse_attention_with_metadata_,
)
from infinicore.ops.deepseek_v4_flashmla_cuda import (
    deepseek_v4_dense_decode_fwd,
    deepseek_v4_dense_decode_fwd_kvfp8,
    deepseek_v4_dense_decode_fwd_qkvfp8,
    deepseek_v4_fwd_kvcache_mla_fp8,
    deepseek_v4_fwd_kvcache_mla_fp8_with_cat,
    deepseek_v4_fwd_kvcache_mla_nope_pe,
    deepseek_v4_fwd_kvcache_quantization_mla,
    deepseek_v4_fwd_kvcache_quantization_q_nope_pe_mla,
    deepseek_v4_get_mla_decoding_metadata_dense_fp8,
    deepseek_v4_sparse_decode_fwd,
    deepseek_v4_sparse_prefill_fwd,
)
from infinicore.ops.deepseek_v4_fused_qk_norm_rope import deepseek_v4_fused_qk_norm_rope, deepseek_v4_fused_qk_norm_rope_
from infinicore.ops.deepseek_v4_fused_rope import deepseek_v4_fused_rope, deepseek_v4_fused_rope_
from infinicore.ops.deepseek_v4_silu_and_mul_clamp import deepseek_v4_silu_and_mul_clamp, deepseek_v4_silu_and_mul_clamp_
from infinicore.ops.deepseek_v4_linear_bf16_fp32 import deepseek_v4_linear_bf16_fp32, deepseek_v4_linear_bf16_fp32_, deepseek_v4_linear_bf16_fp32_blas, deepseek_v4_linear_bf16_fp32_blas_
from infinicore.ops.deepseek_v4_moe_align_block_size import deepseek_v4_moe_align_block_size_
from infinicore.ops.deepseek_v4_moe_marlin_w8a8 import deepseek_v4_moe_marlin_w8a8_, deepseek_v4_moe_marlin_w8a8_fp8_
from infinicore.ops.deepseek_v4_moe_topk_sigmoid import deepseek_v4_moe_topk_sigmoid_
from infinicore.ops.deepseek_v4_moe_topk_softmax import deepseek_v4_moe_topk_softmax_
from infinicore.ops.deepseek_v4_paged_mqa_logits import deepseek_v4_paged_mqa_logits_, deepseek_v4_paged_mqa_logits_metadata_
from infinicore.ops.deepseek_v4_sparse_attn_indexer import deepseek_v4_c4_act_quant_fused_scale_kernel_, deepseek_v4_c4_paged_mqa_logits_, deepseek_v4_c4_sparse_attn_indexer_, deepseek_v4_sparse_attn_indexer_decode_, deepseek_v4_sparse_attn_indexer_prefill_, deepseek_v4_topk_transform_512_kernel_
from infinicore.ops.deepseek_v4_rms_norm import deepseek_v4_rms_norm
from infinicore.ops.deepseek_v4_rmsnorm_self import (
    deepseek_v4_rmsnorm_self,
    deepseek_v4_rmsnorm_self_,
)
from infinicore.ops.deepseek_v4_rms_norm_dynamic_per_token_quant import deepseek_v4_rms_norm_dynamic_per_token_quant_
from infinicore.ops.deepseek_v4_rms_norm_per_block_quant import deepseek_v4_rms_norm_per_block_quant_
from infinicore.ops.deepseek_v4_rotary_embedding import deepseek_v4_rotary_embedding, deepseek_v4_rotary_embedding_
from infinicore.ops.deepseek_v4_silu_and_mul import deepseek_v4_silu_and_mul, deepseek_v4_silu_and_mul_
from infinicore.ops.deepseek_v4_sglang_jit import deepseek_v4_compressed_attn_decode_, deepseek_v4_compressed_attn_metadata_, deepseek_v4_compressed_attn_prefill_, deepseek_v4_flashmla_decode_, deepseek_v4_flashmla_decode_q_nope_pe_, deepseek_v4_flashmla_metadata_, deepseek_v4_flashmla_sparse_prefill_, deepseek_v4_mega_moe_pre_dispatch_, deepseek_v4_silu_and_mul_quant_
from infinicore.ops.deepseek_v4_static_scaled_int8_quant import deepseek_v4_static_scaled_int8_quant_
from infinicore.ops.deepseek_v4_transfer_kv import deepseek_v4_transfer_kv_per_layer_, deepseek_v4_transfer_kv_per_layer_pf_lf_
from infinicore.ops.deepseek_v4_transfer_kv_mla import deepseek_v4_transfer_kv_per_layer_mla_, deepseek_v4_transfer_kv_per_layer_mla_pf_lf_
from infinicore.ops.rearrange import rearrange
from infinicore.ops.reciprocal import reciprocal
from infinicore.ops.rot import rot
from infinicore.ops.rotg import rotg
from infinicore.ops.rotm import rotm
from infinicore.ops.rotmg import rotmg
from infinicore.ops.scal import scal
from infinicore.ops.scatter import scatter
from infinicore.ops.sinh import sinh
from infinicore.ops.squeeze import squeeze
from infinicore.ops.sum import sum
from infinicore.ops.swap import swap
from infinicore.ops.take import take
from infinicore.ops.tan import tan
from infinicore.ops.topk import topk
from infinicore.ops.unsqueeze import unsqueeze
from infinicore.ops.vander import vander
from infinicore.ops.var import var
from infinicore.ops.var_mean import var_mean
from infinicore.tensor import (
    Tensor,
    empty,
    empty_like,
    from_blob,
    from_list,
    from_list_by_numpy,
    from_numpy,
    from_torch,
    ones,
    strided_empty,
    strided_from_blob,
    zeros,
)

__all__ = [
    # Modules.
    "context",
    "nn",
    # Classes.
    "device",
    "DeviceEvent",
    "dtype",
    "Tensor",
    # Context functions.
    "get_device",
    "get_device_count",
    "get_stream",
    "set_device",
    "sync_device",
    "sync_stream",
    "is_graph_recording",
    "start_graph_recording",
    "stop_graph_recording",
    # Data Types.
    "bfloat16",
    "bool",
    "cdouble",
    "cfloat",
    "chalf",
    "complex32",
    "complex64",
    "complex128",
    "double",
    "float",
    "float16",
    "float32",
    "float64",
    "half",
    "int",
    "int8",
    "int16",
    "int32",
    "int64",
    "long",
    "short",
    "uint8",
    # Operations.
    "addcmul",
    "atanh",
    "binary_cross_entropy_with_logits",
    "cdist",
    "reciprocal",
    "add",
    "addr",
    "add_rms_norm",
    "argwhere",
    "asin",
    "asum",
    "axpy",
    "blas_amax",
    "blas_amin",
    "blas_copy",
    "blas_dot",
    "acos",
    "addbmm",
    "floor",
    "attention",
    "mrope",
    "block_diag",
    "kron",
    "bitwise_right_shift",
    "kv_caching",
    "asinh",
    "baddbmm",
    "bilinear",
    "fmod",
    "cat",
    "conv2d",
    "inner",
    "masked_select",
    "logaddexp",
    "logaddexp2",
    "matmul",
    "equal",
    "mul",
    "mul_scalar",
    "diff",
    "digamma",
    "dist",
    "logdet",
    "narrow",
    "nrm2",
    "ldexp",
    "lerp",
    "kthvalue",
    "squeeze",
    "unsqueeze",
    "rearrange",
    "cross_entropy",
    "tan",
    "empty",
    "empty_like",
    "from_blob",
    "from_list",
    "from_list_by_numpy",
    "from_numpy",
    "from_torch",
    "mha_kvcache",
    "mha_varlen",
    "mha",
    "fmin",
    "floor_divide",
    "float_power",
    "flipud",
    "scatter",
    "rot",
    "rotg",
    "rotm",
    "rotmg",
    "scal",
    "logcumsumexp",
    "logical_not",
    "logical_and",
    "vander",
    "paged_caching",
    "paged_attention",
    "paged_attention_prefill",
    "qwen3_add_rms_norm",
    "qwen3_add_rms_norm_inplace",
    "qwen3_fused_qk_norm_rope",
    "qwen3_fused_qk_norm_rope_",
    "qwen3_mha_kvcache",
    "qwen3_mha_varlen",
    "qwen3_rms_norm",
    "qwen3_rotary_embedding",
    "qwen3_rotary_embedding_",
    "qwen3_silu_and_mul",
    "qwen3_store_kvcache",
    "qwen3_store_kvcache_",
    "deepseek_v4_add_rms_norm",
    "deepseek_v4_add_rms_norm_inplace",
    "deepseek_v4_assign_extend_cache_locs_",
    "deepseek_v4_assign_req_to_token_pool_",
    "deepseek_v4_concat_and_cache_mla_",
    "deepseek_v4_create_chunked_prefix_cache_kv_indices_",
    "deepseek_v4_create_flashmla_kv_indices_",
    "deepseek_v4_dcu_alloc_decode_kernel_",
    "deepseek_v4_dcu_alloc_extend_kernel_",
    "deepseek_v4_deep_gemm_low_latency_grouped_gemm_",
    "deepseek_v4_deep_gemm_moe_w8a8_i8_marlin_prefill_down_",
    "deepseek_v4_deep_gemm_moe_w8a8_marlin_decode_down_fp8_",
    "deepseek_v4_dynamic_scaled_int8_quant_",
    "deepseek_v4_fast_topk_",
    "deepseek_v4_fast_topk_transform_fused_",
    "deepseek_v4_fast_topk_transform_ragged_fused_",
    "deepseek_v4_flashmla_cache_indexer_",
    "deepseek_v4_indexer_rotate_",
    "deepseek_v4_fused_experts_impl_int8_marlin_",
    "deepseek_v4_python_fused_experts_impl_int8_marlin_",
    "deepseek_v4_flashmla_sparse_attention_",
    "deepseek_v4_flashmla_sparse_attention_metadata_",
    "deepseek_v4_flashmla_sparse_attention_out_workspace_",
    "deepseek_v4_flashmla_sparse_attention_with_metadata_",
    "deepseek_v4_dense_decode_fwd",
    "deepseek_v4_dense_decode_fwd_kvfp8",
    "deepseek_v4_dense_decode_fwd_qkvfp8",
    "deepseek_v4_fwd_kvcache_mla_fp8",
    "deepseek_v4_fwd_kvcache_mla_fp8_with_cat",
    "deepseek_v4_fwd_kvcache_mla_nope_pe",
    "deepseek_v4_fwd_kvcache_quantization_mla",
    "deepseek_v4_fwd_kvcache_quantization_q_nope_pe_mla",
    "deepseek_v4_get_mla_decoding_metadata_dense_fp8",
    "deepseek_v4_sparse_decode_fwd",
    "deepseek_v4_sparse_prefill_fwd",
    "deepseek_v4_compress_fused_norm_rope_",
    "deepseek_v4_c4_compress_stateful",
    "deepseek_v4_c128_compress_stateful",
    "deepseek_v4_fused_store_flashmla_cache_",
    "deepseek_v4_store_flashmla_raw_cache_",
    "deepseek_v4_store_indexer_raw_cache_",
    "deepseek_v4_fused_qk_norm_rope",
    "deepseek_v4_fused_qk_norm_rope_",
    "deepseek_v4_fused_rope",
    "deepseek_v4_fused_rope_",
    "deepseek_v4_silu_and_mul_clamp",
    "deepseek_v4_silu_and_mul_clamp_",
    "deepseek_v4_linear_bf16_fp32",
    "deepseek_v4_linear_bf16_fp32_",
    "deepseek_v4_linear_bf16_fp32_blas",
    "deepseek_v4_linear_bf16_fp32_blas_",
    "deepseek_v4_moe_align_block_size_",
    "deepseek_v4_moe_marlin_w8a8_",
    "deepseek_v4_moe_marlin_w8a8_fp8_",
    "deepseek_v4_moe_topk_sigmoid_",
    "deepseek_v4_moe_topk_softmax_",
    "deepseek_v4_paged_mqa_logits_",
    "deepseek_v4_paged_mqa_logits_metadata_",
    "deepseek_v4_c4_paged_mqa_logits_",
    "deepseek_v4_c4_sparse_attn_indexer_",
    "deepseek_v4_sparse_attn_indexer_decode_",
    "deepseek_v4_sparse_attn_indexer_prefill_",
    "deepseek_v4_rms_norm",
    "deepseek_v4_rmsnorm_self",
    "deepseek_v4_rmsnorm_self_",
    "deepseek_v4_rms_norm_dynamic_per_token_quant_",
    "deepseek_v4_rms_norm_per_block_quant_",
    "deepseek_v4_rotary_embedding",
    "deepseek_v4_rotary_embedding_",
    "deepseek_v4_silu_and_mul",
    "deepseek_v4_silu_and_mul_",
    "deepseek_v4_silu_and_mul_quant_",
    "deepseek_v4_mega_moe_pre_dispatch_",
    "deepseek_v4_compressed_attn_metadata_",
    "deepseek_v4_compressed_attn_prefill_",
    "deepseek_v4_compressed_attn_decode_",
    "deepseek_v4_flashmla_metadata_",
    "deepseek_v4_flashmla_decode_",
    "deepseek_v4_flashmla_decode_q_nope_pe_",
    "deepseek_v4_flashmla_sparse_prefill_",
    "deepseek_v4_static_scaled_int8_quant_",
    "deepseek_v4_transfer_kv_per_layer_",
    "deepseek_v4_transfer_kv_per_layer_pf_lf_",
    "deepseek_v4_transfer_kv_per_layer_mla_",
    "deepseek_v4_transfer_kv_per_layer_mla_pf_lf_",
    "hypot",
    "index_copy",
    "index_add",
    "take",
    "sinh",
    "swap",
    "ones",
    "broadcast_to",
    "strided_empty",
    "strided_from_blob",
    "zeros",
    "sum",
    "var_mean",
    "moore_mate_flash_attn_prefill",
    "moore_mate_flash_attn_decode",
    "var",
    "topk",
    "all",
    "set_printoptions",
    "printoptions",
]

use_ntops = False

with contextlib.suppress(ImportError, ModuleNotFoundError):
    import sys

    import ntops

    for op_name in ntops.torch.__all__:
        getattr(ntops.torch, op_name).__globals__["torch"] = sys.modules[__name__]

    use_ntops = True
