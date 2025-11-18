# from typing import Union

# import numpy as np

# import infinicore
# from infinicore.nn import functional as F

# from ...tensor import Tensor
# from ..functional import RopeAlgo
# from .module import Module

# __all__ = ["RoPE"]

# # sin_table: (max_position_embeddings, head_dim//2)
# _sin_table: Union[Tensor, None] = None

# # cos_table: (max_position_embeddings, head_dim//2)
# _cos_table: Union[Tensor, None] = None


# def create_sin_cos_table_numpy(max_position, head_dim, theta=10000.0):
#     assert head_dim % 2 == 0, "Embedding dimension must be even."
#     pos = np.arange(0, max_position)
#     freqs = 1.0 / (
#         theta ** (np.arange(0, head_dim, 2)[: (head_dim // 2)].astype(float) / head_dim)
#     )
#     angles = np.outer(pos, freqs)
#     sin_table = np.sin(angles, dtype=np.float32)
#     cos_table = np.cos(angles, dtype=np.float32)
#     return sin_table, cos_table


# def create_sin_cos_table(max_position, head_dim, theta=10000.0):
#     sin_table_np, cos_table_np = create_sin_cos_table_numpy(
#         max_position, head_dim, theta
#     )
#     shape = sin_table_np.shape

#     # ------------------------------------------------ #
#     #    先将numpy 转为一维的 list,再转为 #
#     # ------------------------------------------------ #

#     sin_table_list = sin_table_np.flatten().tolist()
#     cos_table_list = cos_table_np.flatten().tolist()

#     # ------------------------------------------------ #
#     #    再将 list 转为infinicore.Tensor                 #
#     # ------------------------------------------------ #
#     sin_table_infini = infinicore.experimental.convert_list_to_infini_tensor(
#         sin_table_list, shape=shape, infini_dtype=infinicore.float32
#     )
#     cos_table_infini = infinicore.experimental.convert_list_to_infini_tensor(
#         cos_table_list, shape=shape, infini_dtype=infinicore.float32
#     )

#     return sin_table_infini, cos_table_infini


# class RoPE(Module):
#     def __init__(
#         self,
#         max_position_embeddings,
#         rope_theta,
#         head_dim,
#         device=None,
#         dtype=None,
#     ):
#         factory_kwargs = {
#             "device": infinicore.device("cpu", 0) if device is None else device,
#             "dtype": infinicore.float32 if dtype is None else dtype,
#         }
#         super().__init__()

#         self.max_position_embeddings = max_position_embeddings
#         self.rope_theta = rope_theta
#         self.head_dim = head_dim

#         global _sin_table
#         global _cos_table

#         if _sin_table is None:
#             sin_table_infini, cos_table_infini = create_sin_cos_table(
#                 self.max_position_embeddings,
#                 head_dim=self.head_dim,
#                 theta=self.rope_theta,
#             )

#             infini_device = factory_kwargs["device"]

#             _sin_table = sin_table_infini.to(infini_device)
#             _cos_table = cos_table_infini.to(infini_device)

#             # _sin_table = sin_table_infini.to(infinicore.device("cpu", 0))
#             # _cos_table = cos_table_infini.to(infinicore.device("cpu", 0))

#     def forward(
#         self,
#         states: Tensor,
#         position_ids: Tensor,
#         algo=RopeAlgo.GPT_NEOX,
#     ):
#         bs, ntok, num_attention_heads, head_dim = states.shape
#         states = states.view((bs * ntok, num_attention_heads, head_dim))

#         if algo == RopeAlgo.GPT_J:
#             raise ValueError(" Algorithm.GPT_J not support")
#         else:
#             out = F.rope(states, position_ids, _sin_table, _cos_table)
#         out_torch = out.view((bs, ntok, num_attention_heads, head_dim))
#         return out_torch


from typing import Union

import numpy as np

import infinicore
from infinicore.nn import functional as F

from ...tensor import Tensor
from ..functional import RopeAlgo
from .module import InfiniCoreModule as Module

# # sin_table: (max_position_embeddings, head_dim//2)
# _sin_table: Union[Tensor, None] = None

# # cos_table: (max_position_embeddings, head_dim//2)
# _cos_table: Union[Tensor, None] = None


def create_sin_cos_table_numpy(max_position, head_dim, theta=10000.0):
    assert head_dim % 2 == 0, "Embedding dimension must be even."
    pos = np.arange(0, max_position)
    freqs = 1.0 / (
        theta ** (np.arange(0, head_dim, 2)[: (head_dim // 2)].astype(float) / head_dim)
    )
    angles = np.outer(pos, freqs)
    sin_table = np.sin(angles, dtype=np.float32)
    cos_table = np.cos(angles, dtype=np.float32)
    return sin_table, cos_table


def create_sin_cos_table(
    max_position,
    head_dim,
    theta=10000.0,
    device=None,
    dtype=None,
):
    # device = (infinicore.device("cpu", 0),)
    # dtype=infinicore.float32,

    sin_table_np, cos_table_np = create_sin_cos_table_numpy(
        max_position, head_dim, theta
    )

    sin_table_infini = infinicore.from_numpy(sin_table_np, dtype=dtype, device=device)
    cos_table_infini = infinicore.from_numpy(cos_table_np, dtype=dtype, device=device)

    return sin_table_infini, cos_table_infini


class RoPE(Module):
    def __init__(
        self,
        max_position_embeddings: int,
        rope_theta: float,
        head_dim: int,
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

        self._sin_table, self._cos_table = create_sin_cos_table(
            self.max_position_embeddings,
            head_dim=self.head_dim,
            theta=self.rope_theta,
            **factory_kwargs,
        )

    def forward(
        self,
        states: Tensor,  #  bs, seq_len, num_attention_heads, head_dim
        position_ids: Tensor,  # bs,num
        algo=RopeAlgo.GPT_NEOX,
    ):
        bs, seq_len, num_attention_heads, head_dim = states.shape

        for i in range(0, bs):
            state_i = states.narrow(0, i, 1).view(
                (1 * seq_len, num_attention_heads, head_dim)
            )
            F.rope(
                state_i,
                position_ids,
                self._sin_table,
                self._cos_table,
                algo=algo,
                out=state_i,
            )

        return states
