import torch
import infinicore
from typing import Union
from .module import Module

class SiLU(Module):
    r"""
    Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
    """
    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward_torch(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(input, inplace=self.inplace)

    def forward_infini(self, input: infinicore.Tensor) -> infinicore.Tensor:
        return infinicore.nn.functional.silu(input, inplace=self.inplace)

    def forward(self,
                input: Union[infinicore.Tensor, torch.Tensor]
                ) -> Union[infinicore.Tensor, torch.Tensor]:
        if isinstance(input, torch.Tensor):
            return self.forward_torch(input)

        return self.forward_infini(input)

    def extra_repr(self) -> str:
        return " infinicore op : inplace=True" if self.inplace else ""



class SwiGLU(Module):
    r"""
    Apply the Swish-Gated Linear Unit (SwiGLU) function, element-wise.
    """
    def __init__(self):
        super().__init__()
 
    def forward_torch(self, 
                      input: torch.Tensor,
                       other: torch.Tensor,
                      ) -> torch.Tensor:
        raise KeyError("not support")
        return torch.nn.functional.relu(input, inplace=self.inplace)

    def forward_infini(self, 
                            input: infinicore.Tensor,
                            other: infinicore.Tensor) -> infinicore.Tensor:
        return infinicore.nn.functional.swiglu(input, other)

    def forward(self,
                input: Union[infinicore.Tensor, torch.Tensor],
                other: Union[infinicore.Tensor, torch.Tensor]
                ) -> Union[infinicore.Tensor, torch.Tensor]:
        
        if isinstance(input, torch.Tensor):
            return self.forward_torch(input,other)

        return self.forward_infini(input,other)

    def extra_repr(self) -> str:
        return " infinicore op" 
    

