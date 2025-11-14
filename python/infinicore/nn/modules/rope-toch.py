from enum import Enum
from typing import Union

import torch

import infinicore
from infinicore.nn import functional as F

from ...tensor import Tensor
from .module import Module

__all__ = ["RoPE"]

# sin_table: (max_position_embeddings, head_dim//2)
_sin_table: Union[Tensor, None] = None

# cos_table: (max_position_embeddings, head_dim//2)
_cos_table: Union[Tensor, None] = None


class Algorithm(Enum):
    GPT_J = 0
    GPT_NEOX = 1


def create_sin_cos_table(
    max_position, head_dim, theta=10000.0, torch_dtype=torch.float32, torch_device="cpu"
):
    assert head_dim % 2 == 0, "Embedding dimension must be even."
    pos = torch.arange(0, max_position)
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim)
    )
    angles = torch.outer(pos, freqs)
    return torch.sin(angles).to(dtype=torch_dtype, device=torch_device), torch.cos(
        angles
    ).to(dtype=torch_dtype, device=torch_device)


class RoPE(Module):  # Module
    def __init__(
        self,
        max_position_embeddings,
        rope_theta,
        head_dim,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {
            "device": infinicore.device("cpu", 0) if device is None else device,
            "dtype": infinicore.float32 if dtype is None else dtype,
        }
        super().__init__()

        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.head_dim = head_dim

    def forward(
        self,
        states: Tensor,
        position_ids: Tensor,
        algo=Algorithm.GPT_NEOX,
    ):
        global _sin_table
        global _cos_table
        if _sin_table is None:
            dtype = infinicore.to_torch_dtype(states.dtype)
            device = states.device.type

            sin_table, cos_table = create_sin_cos_table(
                self.max_position_embeddings,
                head_dim=self.head_dim,
                theta=self.rope_theta,
                torch_dtype=dtype,
                torch_device=device,
            )
            _sin_table = infinicore.convert_torch_to_infini_tensor(sin_table)
            _cos_table = infinicore.convert_torch_to_infini_tensor(cos_table)

        bs, ntok, num_attention_heads, head_dim = states.shape
        states = states.view((bs * ntok, num_attention_heads, head_dim))

        if algo == Algorithm.GPT_J:
            raise ValueError(" Algorithm.GPT_J not support")
        else:
            out = F.rope(states, position_ids, _sin_table, _cos_table)
        out_torch = out.view((bs, ntok, num_attention_heads, head_dim))
        return out_torch
