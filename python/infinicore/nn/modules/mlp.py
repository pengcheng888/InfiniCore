import torch
import infinicore
from typing import Union
from .module import Module
from .linear import Linear
from .activation import SiLU


class MLP(Module):
    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 mlp_bias: bool,
                 ):
        '''
        LlamaMLP
        '''
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
        self.up_proj = Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
        self.down_proj = Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias)
        self.act_fn = SiLU()

    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        print('MLP forward_torch')
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

    def forward_infinicore(self, x: infinicore.Tensor):
        print('MLP forward_infinicore')
        raise Exception('RMSNorm forward_infinicore not support !!!')

    def forward(self,
                x: Union[infinicore.Tensor, torch.Tensor]
                ) -> Union[infinicore.Tensor, torch.Tensor]:
        if isinstance(x, torch.Tensor):
            return self.forward_torch(x)
        return self.forward_infinicore(x)

    def extra_repr(self):
        return f" infinicore op : MLP"

    @staticmethod
    def testop():
        pass
