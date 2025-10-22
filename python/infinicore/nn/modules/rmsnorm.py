import torch
import infinicore
from typing import Union
from .module import Module

class RMSNorm(Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward_torch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        print('RMSNorm forward_torch')
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def forward_infinicore(self, hidden_states: infinicore.Tensor):
        print('RMSNorm forward_infinicore')
        raise Exception('RMSNorm forward_infinicore not support !!!')

    def forward(self,
                hidden_states: Union[infinicore.Tensor, torch.Tensor]
                ) -> Union[infinicore.Tensor, torch.Tensor]:
        if isinstance(hidden_states, torch.Tensor):
            return self.forward_torch(hidden_states)
        return self.forward_infinicore(hidden_states)

    def extra_repr(self):
        return f" infinicore op : {tuple(self.weight.shape)}, eps={self.variance_epsilon}"

    @staticmethod
    def testop():
        hidden_size = 8
        norm = RMSNorm(hidden_size)

        x = torch.rand((5, 8), dtype=torch.float32)
        print(x)
        y = norm(x)
        print(y)
