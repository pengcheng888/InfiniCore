from typing import Optional

import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from typing import Union
from .module import Module

__all__ = ["Embedding"]

import infinicore
from infinicore.nn.modules.linear import create_infinicore_tensor, infini_tensor_2_torch_tensor


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

    Examples::

        >>> # an Embedding module containing 10 tensors of size 3
        >>> embedding = nn.Embedding(10, 3)
        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> embedding(input)
        tensor([[[-0.0251, -1.6902,  0.7172],
                 [-0.6431,  0.0748,  0.6969],
                 [ 1.4970,  1.3448, -0.9685],
                 [-0.3677, -2.7265, -0.1685]],

                [[ 1.4970,  1.3448, -0.9685],
                 [ 0.4362, -0.4004,  0.9400],
                 [-0.6431,  0.0748,  0.6969],
                 [ 0.9124, -2.3616,  1.1151]]])
    """

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
    ]

    num_embeddings: int
    embedding_dim: int
    weight: torch.Tensor

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            _weight: Optional[torch.Tensor] = None,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight_infini = None
        if _weight is None:
            self.weight = Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs), requires_grad=False)
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim, ], "Shape of weight does not match num_embeddings and embedding_dim"
            self.weight = Parameter(_weight, requires_grad=False)

    def forward(self,
                input: Union[infinicore.Tensor, torch.Tensor]
                ) -> Union[infinicore.Tensor, torch.Tensor]:
        if isinstance(input, torch.Tensor):
            return self.forward_torch2torch(input)

        # self.forward_infini2infini(input)
        # self.forward_torch2infini2torch(input)
        return self.forward_infini2infini(input)

    def forward_infini2infini(self, input: infinicore.Tensor) -> infinicore.Tensor:
        device_str = "cpu"
        if self.weight_infini is None:
            self.weight_infini = create_infinicore_tensor(self.weight, device_str)
        return infinicore.embedding(input, self.weight_infini)

    def forward_torch2torch(self,
                            input: torch.Tensor
                            ) -> torch.Tensor:
        return F.embedding(input, self.weight)

    def forward_torch2infini2torch(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(dtype=torch.int32)

        import infinicore
        from infinicore.nn.modules.linear import create_infinicore_tensor, infini_tensor_2_torch_tensor

        device_str = "cpu"
        input_infini = create_infinicore_tensor(input, device_str)
        weight_infini = create_infinicore_tensor(self.weight, device_str)

        inputs_embeds_infini = infinicore.embedding(input_infini, weight_infini)
        inputs_embeds_torch = infini_tensor_2_torch_tensor(inputs_embeds_infini)

        return inputs_embeds_torch

    def extra_repr(self) -> str:
        s = "{num_embeddings}, {embedding_dim}"
        return s.format(**self.__dict__)

    @classmethod
    def from_pretrained(
            cls,
            embeddings,
    ):
        r"""Create Embedding instance from given 2-dimensional FloatTensor.

        Args:
            embeddings (Tensor): FloatTensor containing weights for the Embedding.
                First dimension is being passed to Embedding as ``num_embeddings``, second as ``embedding_dim``.

        Examples::

            >>> # FloatTensor containing pretrained weights
            >>> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
            >>> embedding = nn.Embedding.from_pretrained(weight)
            >>> # Get embeddings for index 1
            >>> input = torch.LongTensor([1])
            >>> # xdoctest: +IGNORE_WANT("non-deterministic")
            >>> embedding(input)
            tensor([[ 4.0000,  5.1000,  6.3000]])
        """

        assert (embeddings.dim() == 2), "Embeddings parameter is expected to be 2-dimensional"
        rows, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
        )
        return embedding
