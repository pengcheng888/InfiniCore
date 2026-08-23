from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def deepseek_v4_fused_rope_(
    query: Tensor,
    key: Tensor | None,
    freqs_cis: Tensor,
    positions: Tensor,
    inverse: bool = False,
) -> tuple[Tensor, Tensor | None]:
    _infinicore.deepseek_v4_fused_rope_(
        query._underlying,
        None if key is None else key._underlying,
        freqs_cis._underlying,
        positions._underlying,
        inverse,
    )
    return query, key


def deepseek_v4_fused_rope(
    query: Tensor,
    key: Tensor | None,
    freqs_cis: Tensor,
    positions: Tensor,
    inverse: bool = False,
) -> tuple[Tensor, Tensor | None]:
    return deepseek_v4_fused_rope_(query, key, freqs_cis, positions, inverse)
