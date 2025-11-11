from typing import Optional

import torch

from typing import Union
from .module import Module

__all__ = ["Embedding"]

import infinicore

class Embedding(Module):
    r"""A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector

    Attributes:
        weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)
                         initialized from :math:`\mathcal{N}(0, 1)`

    Shape:
        - Input: :math:`(*)`, IntTensor or LongTensor of arbitrary shape containing the indices to extract
        - Output: :math:`(*, H)`, where `*` is the input shape and :math:`H=\text{embedding\_dim}`
    """

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
    ]

    num_embeddings: int
    embedding_dim: int
    weight: infinicore.Tensor

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
     
        self.weight = infinicore.nn.Parameter(
            infinicore.empty((1,), dtype=infinicore.float32, device=infinicore.device("cpu", 0))
            )

    def forward(self, input: infinicore.Tensor) -> infinicore.Tensor:
        return  infinicore.nn.functional.embedding(input, self.weight)


    def extra_repr(self) -> str:
        s = "{num_embeddings}, {embedding_dim}"
        return s.format(**self.__dict__)
    

# class Embedding(Module):
#     r"""A simple lookup table that stores embeddings of a fixed dictionary and size.

#     This module is often used to store word embeddings and retrieve them using indices.
#     The input to the module is a list of indices, and the output is the corresponding
#     word embeddings.

#     Args:
#         num_embeddings (int): size of the dictionary of embeddings
#         embedding_dim (int): the size of each embedding vector

#     Attributes:
#         weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)
#                          initialized from :math:`\mathcal{N}(0, 1)`

#     Shape:
#         - Input: :math:`(*)`, IntTensor or LongTensor of arbitrary shape containing the indices to extract
#         - Output: :math:`(*, H)`, where `*` is the input shape and :math:`H=\text{embedding\_dim}`
#     """

#     __constants__ = [
#         "num_embeddings",
#         "embedding_dim",
#     ]

#     num_embeddings: int
#     embedding_dim: int
#     weight: torch.Tensor

#     def __init__(
#             self,
#             num_embeddings: int,
#             embedding_dim: int,
#             _weight: Optional[torch.Tensor] = None,
#             device=None,
#             dtype=None,
#     ) -> None:
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         self.num_embeddings = num_embeddings
#         self.embedding_dim = embedding_dim
#         self.weight_infini = None
#         if _weight is None:
#             self.weight = Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs), requires_grad=False)
#         else:
#             assert list(_weight.shape) == [num_embeddings,
#                                            embedding_dim, ], "Shape of weight does not match num_embeddings and embedding_dim"
#             self.weight = Parameter(_weight, requires_grad=False)

#     def forward(self,
#                 input: Union[infinicore.Tensor, torch.Tensor]
#                 ) -> Union[infinicore.Tensor, torch.Tensor]:
#         if isinstance(input, torch.Tensor):
#             return self.forward_torch2torch(input)

#         return self.forward_infini2infini(input)

#     def forward_infini2infini(self, input: infinicore.Tensor) -> infinicore.Tensor:
#         if self.weight_infini is None:
#             self.weight_infini =  infinicore.convert_torch_to_infini_tensor(self.weight)

   
#         return  infinicore.nn.functional.embedding(input, self.weight_infini)

#     def forward_torch2torch(self,
#                             input: torch.Tensor
#                             ) -> torch.Tensor:
#         return torch.nn.functional.embedding(input, self.weight)

#     def extra_repr(self) -> str:
#         s = "{num_embeddings}, {embedding_dim}"
#         return s.format(**self.__dict__)