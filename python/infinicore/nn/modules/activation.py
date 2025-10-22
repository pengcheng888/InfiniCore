import torch
import infinicore
from typing import Union
from .module import Module
from torch.nn import functional as F


class SiLU(Module):
    r"""
    Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
    """
    # __constants__是一个特殊的属性，用于告诉PyTorch哪些属性应该被视为常量，从而在编译时可以更好地优化
    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward_torch(self, input: torch.Tensor) -> torch.Tensor:
        print('SiLU forward_torch')
        return F.silu(input, inplace=self.inplace)

    def forward_infinicore(self, input: infinicore.Tensor):
        print('SiLU forward_infinicore')
        raise Exception('SiLU forward_infinicore not support !!!')

    def forward(self,
                input: Union[infinicore.Tensor, torch.Tensor]
                ) -> Union[infinicore.Tensor, torch.Tensor]:
        if isinstance(input, torch.Tensor):
            return self.forward_torch(input)
        return self.forward_infinicore(input)

    def extra_repr(self) -> str:
        return " infinicore op : inplace=True" if self.inplace else ""

    @staticmethod
    def testop():
        m = SiLU()
        input = torch.randn(2)
        output = m(input)
