from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def deepseek_v4_flashmla_sparse_attention_(
    q: Tensor,
    raw_cache: Tensor,
    indices: Tensor,
    topk_lengths: Tensor,
    attn_sink: Tensor | None,
    output: Tensor,
    softmax_scale: float,
    page_size: int = 256,
    head_dim_v: int = 512,
    extra_raw_cache: Tensor | None = None,
    extra_indices: Tensor | None = None,
    extra_topk_lengths: Tensor | None = None,
    extra_page_size: int = 0,
) -> Tensor:
    _infinicore.deepseek_v4_flashmla_sparse_attention_(
        q._underlying,
        raw_cache._underlying,
        indices._underlying,
        topk_lengths._underlying,
        None if attn_sink is None else attn_sink._underlying,
        output._underlying,
        softmax_scale,
        page_size,
        head_dim_v,
        None if extra_raw_cache is None else extra_raw_cache._underlying,
        None if extra_indices is None else extra_indices._underlying,
        None if extra_topk_lengths is None else extra_topk_lengths._underlying,
        extra_page_size,
    )
    return output
