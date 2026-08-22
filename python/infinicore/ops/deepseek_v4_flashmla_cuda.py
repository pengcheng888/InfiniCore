from infinicore.lib import _infinicore


def flash_mla_dense_decode_fwd(*args, **kwargs):
    return _infinicore.flash_mla_dense_decode_fwd(*args, **kwargs)


def deepseek_v4_dense_decode_fwd_kvfp8(*args, **kwargs):
    return _infinicore.deepseek_v4_dense_decode_fwd_kvfp8(*args, **kwargs)


def deepseek_v4_dense_decode_fwd_qkvfp8(*args, **kwargs):
    return _infinicore.deepseek_v4_dense_decode_fwd_qkvfp8(*args, **kwargs)


def deepseek_v4_fwd_kvcache_mla_fp8(*args, **kwargs):
    return _infinicore.deepseek_v4_fwd_kvcache_mla_fp8(*args, **kwargs)


def deepseek_v4_fwd_kvcache_mla_fp8_with_cat(*args, **kwargs):
    return _infinicore.deepseek_v4_fwd_kvcache_mla_fp8_with_cat(*args, **kwargs)


def deepseek_v4_fwd_kvcache_mla_nope_pe(*args, **kwargs):
    return _infinicore.deepseek_v4_fwd_kvcache_mla_nope_pe(*args, **kwargs)


def deepseek_v4_fwd_kvcache_quantization_mla(*args, **kwargs):
    return _infinicore.deepseek_v4_fwd_kvcache_quantization_mla(*args, **kwargs)


def deepseek_v4_fwd_kvcache_quantization_q_nope_pe_mla(*args, **kwargs):
    return _infinicore.deepseek_v4_fwd_kvcache_quantization_q_nope_pe_mla(*args, **kwargs)


def deepseek_v4_get_mla_decoding_metadata_dense_fp8(*args, **kwargs):
    return _infinicore.deepseek_v4_get_mla_decoding_metadata_dense_fp8(*args, **kwargs)


def flash_mla_sparse_decode_fwd(*args, **kwargs):
    return _infinicore.flash_mla_sparse_decode_fwd(*args, **kwargs)


def deepseek_v4_sparse_prefill_fwd(*args, **kwargs):
    return _infinicore.deepseek_v4_sparse_prefill_fwd(*args, **kwargs)
