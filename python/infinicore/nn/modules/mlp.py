import torch
import infinicore
from typing import Union
from .module import Module
from .linear import Linear
from .activation import InfiniSiLU as SiLU


class MLP(Module):
    def __init__(self, config):
        super().__init__()

        import infinicore

        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = infinicore.nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = infinicore.nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = infinicore.nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = SiLU()  # ACT2FN[config.hidden_act]

    def forward(self, x: Union[infinicore.Tensor, torch.Tensor]) -> Union[infinicore.Tensor, torch.Tensor]:
        if isinstance(x, torch.Tensor):
            return self.forward_torch2torch(x)
        return self.forward_infini2infini(x)

    def forward_infini2infini(self, x: infinicore.Tensor) -> infinicore.Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

    def forward_torch2torch(self, x: torch.Tensor):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

    def extra_repr(self):
        return f" infinicore op : MLP"
