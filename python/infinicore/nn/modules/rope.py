import torch
import infinicore
from typing import Union
from .module import Module
from typing import Union
from enum import Enum, auto

_sin_table: Union[infinicore.Tensor, None] = None  # sin_table: (max_position_embeddings, head_dim//2)
_cos_table: Union[infinicore.Tensor, None] = None  # cos_table: (max_position_embeddings, head_dim//2)


class Algorithm(Enum):
    GPT_J = 0
    GPT_NEOX = 1


def create_sin_cos_table(max_position, head_dim=64, theta=10000.0):
    assert head_dim % 2 == 0, "Embedding dimension must be even."
    pos = torch.arange(0, max_position)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
    angles = torch.outer(pos, freqs)
    return torch.sin(angles), torch.cos(angles)


class RoPE():  # Module
    def __init__(self, config):
        global _sin_table
        global _cos_table

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        if _sin_table is None:
            sin_table, cos_table = create_sin_cos_table(self.max_position_embeddings, head_dim=self.head_dim,
                                                        theta=self.rope_theta)

            from infinicore.nn.modules.linear import create_infinicore_tensor, infini_tensor_2_torch_tensor
            device_str = "cpu"
            _sin_table = create_infinicore_tensor(sin_table.to(dtype=torch.float16), device_str)
            _cos_table = create_infinicore_tensor(cos_table.to(dtype=torch.float16), device_str)

        print("create _sin_table", id(_sin_table))

    def forward(self,
                states: Union[infinicore.Tensor, torch.Tensor],
                position_ids: torch.Tensor):
        if isinstance(input, torch.Tensor):
            self.forward_torch2infini2torch(states, position_ids)

        return self.forward_infini2infini(states, position_ids)

    def forward_torch2infini2torch(self, states: torch.Tensor,
                                   position_ids: torch.Tensor,
                                   algo=Algorithm.GPT_NEOX):

        # print("RoPE:: forward_torch", id(_sin_table))
        from infinicore.nn.modules.linear import torch_tensor_2_infini_tensor, infini_tensor_2_torch_tensor
        bs, num_attention_heads, ntok, head_dim = states.shape

        states = states.transpose(1, 2).reshape(-1, num_attention_heads, head_dim).contiguous()

        if algo == Algorithm.GPT_J:
            raise ValueError("RoPE_infinicore not support")
        else:
            out = infinicore.rope(torch_tensor_2_infini_tensor(states),
                                  torch_tensor_2_infini_tensor(position_ids),
                                  _sin_table,
                                  _cos_table,
                                  )
        out_torch = infini_tensor_2_torch_tensor(out)
        out_torch = out_torch.reshape(-1, ntok, num_attention_heads, head_dim).transpose(1, 2).contiguous()
        return out_torch

    def forward_infini2infini(self, states: infinicore.Tensor,
                              position_ids: torch.Tensor,
                              algo=Algorithm.GPT_NEOX):

        # print("RoPE:: forward_infini ", id(_sin_table))

        from infinicore.nn.modules.linear import torch_tensor_2_infini_tensor
        bs, ntok, num_attention_heads, head_dim = states.shape

        states = states.view((bs * ntok, num_attention_heads, head_dim))

        if algo == Algorithm.GPT_J:
            raise ValueError("RoPE not support")
        else:
            out = infinicore.rope(states,
                                  torch_tensor_2_infini_tensor(position_ids),
                                  _sin_table,
                                  _cos_table)

        out_torch = out.view((bs, ntok, num_attention_heads, head_dim))
        return out_torch
