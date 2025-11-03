import torch
import infinicore
from typing import Union
from .module import Module
from torch.nn import functional as F


class InfiniSiLU(Module):
    r"""
    Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
    """
    # __constants__是一个特殊的属性，用于告诉PyTorch哪些属性应该被视为常量，从而在编译时可以更好地优化
    __constants__ = ["inplace"]
    inplace: bool
    shared_count = 0

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward_torch(self, input: torch.Tensor) -> torch.Tensor:
        InfiniSiLU.shared_count += 1
        # print('InfiniSiLU forward_torch ',  InfiniSiLU.shared_count )
        return F.silu(input, inplace=self.inplace)

    def forward_infinicore(self, input: infinicore.Tensor, device_str="cpu"):
        InfiniSiLU.shared_count += 1
        # print('InfiniSiLU forward_infinicore ', InfiniSiLU.shared_count )

        # 转为 torch 计算
        from .linear import create_infinicore_tensor, infini_tensor_2_torch_tensor

        input_torch = infini_tensor_2_torch_tensor(input, device_str=device_str)
        output_torch = F.silu(input_torch, inplace=self.inplace)
        output_infinicore = create_infinicore_tensor(output_torch, device_str)

        # raise Exception('SiLU forward_infinicore not support !!!')
        return output_infinicore

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
