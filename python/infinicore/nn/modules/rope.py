import torch
import infinicore
from typing import Union
from .module import Module
from typing import Union
from enum import Enum, auto


__all__ = [  "RoPE" ]

_sin_table: Union[infinicore.Tensor, None] = None  # sin_table: (max_position_embeddings, head_dim//2)
_cos_table: Union[infinicore.Tensor, None] = None  # cos_table: (max_position_embeddings, head_dim//2)


class Algorithm(Enum):
    GPT_J = 0
    GPT_NEOX = 1


def create_sin_cos_table(max_position, 
                         head_dim, 
                         theta=10000.0, 
                         torch_dtype = torch.float32,
                         torch_device = "cpu"):
    
    assert head_dim % 2 == 0, "Embedding dimension must be even."
    pos = torch.arange(0, max_position)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
    angles = torch.outer(pos, freqs)
    return torch.sin(angles).to(dtype=torch_dtype, device=torch_device), torch.cos(angles).to(dtype=torch_dtype, device=torch_device)


class RoPE():  # Module
    def __init__(self, config):


        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads



    def forward(self, states: infinicore.Tensor,
                              position_ids: infinicore.Tensor,
                              algo=Algorithm.GPT_NEOX
                              ):
        global _sin_table
        global _cos_table
        if _sin_table is None:
            dtype= infinicore.to_torch_dtype(states.dtype)     
            device = states.device.type
    
            sin_table, cos_table = create_sin_cos_table(self.max_position_embeddings, 
                                                        head_dim=self.head_dim,
                                                        theta=self.rope_theta,
                                                        torch_dtype=dtype,
                                                        torch_device =device
                                                        )
            _sin_table =  infinicore.convert_torch_to_infini_tensor(sin_table)
            _cos_table =  infinicore.convert_torch_to_infini_tensor(cos_table)

            # torch.cuda.synchronize()

        bs, ntok, num_attention_heads, head_dim = states.shape
        states = states.view((bs * ntok, num_attention_heads, head_dim))

        if algo == Algorithm.GPT_J:
            raise ValueError(" Algorithm.GPT_J not support")
        else:
            out = infinicore.nn.functional.rope(states,
                                   position_ids,
                                  _sin_table,
                                  _cos_table)
        out_torch = out.view((bs, ntok, num_attention_heads, head_dim))
        return out_torch
    

# class RoPE():  # Module
#     def __init__(self, config):
#         global _sin_table
#         global _cos_table

#         self.max_position_embeddings = config.max_position_embeddings
#         self.rope_theta = config.rope_theta
#         self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

#         if _sin_table is None:
#             sin_table, cos_table = create_sin_cos_table(self.max_position_embeddings, 
#                                                         head_dim=self.head_dim,
#                                                         theta=self.rope_theta)
#             _sin_table =  infinicore.convert_torch_to_infini_tensor(sin_table.to(dtype=torch.float16))
#             _cos_table =  infinicore.convert_torch_to_infini_tensor(cos_table.to(dtype=torch.float16))

#     def forward(self,
#                 states: Union[infinicore.Tensor, torch.Tensor],
#                 position_ids: torch.Tensor):
#         if isinstance(input, torch.Tensor):
#             self.forward_torch2infini2torch(states, position_ids)

#         return self.forward_infini2infini(states, position_ids)

#     def forward_torch2infini2torch(self, states: torch.Tensor,
#                                    position_ids: torch.Tensor,
#                                    algo=Algorithm.GPT_NEOX):

#         # print("RoPE:: forward_torch", id(_sin_table))

#         bs, num_attention_heads, ntok, head_dim = states.shape

#         states = states.transpose(1, 2).reshape(-1, num_attention_heads, head_dim).contiguous()

#         if algo == Algorithm.GPT_J:
#             raise ValueError("RoPE_infinicore not support")
#         else:
             
#             out = infinicore.rope( infinicore.convert_torch_to_infini_tensor(states),
#                                    infinicore.convert_torch_to_infini_tensor(position_ids),
#                                   _sin_table,
#                                   _cos_table,
#                                   )
#         out_torch =   infinicore.convert_infini_to_torch_tensor(out)
#         out_torch = out_torch.reshape(-1, ntok, num_attention_heads, head_dim).transpose(1, 2).contiguous()
#         return out_torch

#     def forward_infini2infini(self, states: infinicore.Tensor,
#                               position_ids: torch.Tensor,
#                               algo=Algorithm.GPT_NEOX):

 
#         bs, ntok, num_attention_heads, head_dim = states.shape
#         states = states.view((bs * ntok, num_attention_heads, head_dim))

#         if algo == Algorithm.GPT_J:
#             raise ValueError(" Algorithm.GPT_J not support")
#         else:
#             out = infinicore.nn.functional.rope(states,
#                                    infinicore.convert_torch_to_infini_tensor(position_ids),
#                                   _sin_table,
#                                   _cos_table)
#         out_torch = out.view((bs, ntok, num_attention_heads, head_dim))
#         return out_torch