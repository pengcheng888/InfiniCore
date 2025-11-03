import torch
import infinicore
from typing import Union
from .module import Module


# class RMSNorm(Module):
#     def __init__(self, hidden_size, eps=1e-6):
#         """
#         LlamaRMSNorm is equivalent to LlamaRMSNorm
#         """
#         super().__init__()
#         self.weight = torch.nn.Parameter(torch.ones(hidden_size))
#         self.variance_epsilon = eps
#         self.weight_infini = None

#     def forward(self,
#                 hidden_states: Union[infinicore.Tensor, torch.Tensor]
#                 ) -> Union[infinicore.Tensor, torch.Tensor]:
#         if isinstance(hidden_states, infinicore.Tensor):
#             return self.forward_infini2infini(hidden_states)

#         return self.forward_torch2torch(hidden_states)

#     def forward_torch2torch(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         # print(" LlamaRMSNorm :: forward_torch2torch ")
#         input_dtype = hidden_states.dtype
#         hidden_states = hidden_states.to(torch.float32)
#         variance = hidden_states.pow(2).mean(-1, keepdim=True)
#         hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
#         return self.weight * hidden_states.to(input_dtype)

#     def forward_infini2infini(self, hidden_states: infinicore.Tensor) -> infinicore.Tensor:
#         from infinicore.nn.modules.linear import torch_tensor_2_infini_tensor

#         # print(" LlamaRMSNorm :: forward_infini2infini ")
#         if self.weight_infini is None:
#             self.weight_infini = torch_tensor_2_infini_tensor(self.weight)
#         return infinicore.rms_norm(hidden_states, self.weight_infini, self.variance_epsilon)

#     def extra_repr(self):
#         return f" infinicore op : {tuple(self.weight.shape)}, eps={self.variance_epsilon}"
