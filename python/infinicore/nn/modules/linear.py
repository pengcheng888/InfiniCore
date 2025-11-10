import torch
import infinicore
from typing import Union
from .module import Module

class Linear(Module):
    r"""Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_\text{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_\text{in} = \text{in\_features}`.
        - Output: :math:`(*, H_\text{out})` where all but the last dimension
          are the same shape as the input and :math:`H_\text{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = infinicore.nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = infinicore.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.weight_infini = None
        self.bias_infini = None

    def forward(self,
                input: Union[infinicore.Tensor, torch.Tensor],
                ) -> Union[infinicore.Tensor, torch.Tensor]:
        if isinstance(input, torch.Tensor):
            return self.forward_torch(input)

        return self.forward_infini(input)

    def forward_torch(self, input:  torch.Tensor) ->  torch.Tensor:

        # in_features 是 20
        # out_features 是 30
        # 将输入的维度从 in_features 变成 out_features
        # F.linear(input, self.weight, self.bias) # size([128,in_features]),size([out_features,in_features]) ==> size([128,out_features])
        return  torch.nn.functional.linear(input, self.weight, self.bias)
    
    def forward_infini(self, input: infinicore.Tensor) ->  infinicore.Tensor:

        # in_features 是 20
        # out_features 是 30
        # 将输入的维度从 in_features 变成 out_features
        # F.linear(input, self.weight, self.bias) # size([128,in_features]),size([out_features,in_features]) ==> size([128,out_features])
        if self.weight_infini is None:
            self.weight_infini = infinicore.convert_torch_to_infini_tensor(self.weight)
        if (self.bias_infini is None) and (self.bias is not None):
            self.bias_infini = infinicore.convert_torch_to_infini_tensor(self.bias)

        return  infinicore.nn.functional.linear(input, self.weight_infini, self.bias_infini)
    
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"