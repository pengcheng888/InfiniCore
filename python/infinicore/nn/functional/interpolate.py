from typing import Optional, Sequence, Union

from infinicore.tensor import Tensor

from .upsample_bilinear import upsample_bilinear
from .upsample_nearest import upsample_nearest


def interpolate(
    input: Tensor,
    size: Optional[Union[int, Sequence[int]]] = None,
    scale_factor: Optional[Union[float, Sequence[float]]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
) -> Tensor:
    if mode == "nearest":
        if align_corners is not None:
            raise ValueError(
                "align_corners option can only be set with the "
                "interpolating modes: linear | bilinear | bicubic | trilinear"
            )
        return upsample_nearest(input, size, scale_factor)

    if mode == "bilinear":
        if align_corners is None:
            align_corners = False
        return upsample_bilinear(input, size, scale_factor, align_corners)

    raise NotImplementedError(
        f"Interpolation mode '{mode}' is not currently supported."
    )
