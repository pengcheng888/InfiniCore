from infinicore.lib import _infinicore
from infinicore.tensor import Tensor
from typing import Union


def embedding(input: Tensor,  # LongTensor of arbitrary shape containing the indices to extract
              weight: Tensor,  # Weight: Embedding matrix of floating point type with shape (V, embedding_dim), where V = maximum index + 1
              ) -> Tensor:
    return Tensor(_infinicore.embedding(input._underlying, weight._underlying))
