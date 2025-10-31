from infinicore.lib import _infinicore
from infinicore.tensor import Tensor
from typing import Union
import math

def linear(input: Tensor,  # (∗,in_features) where * means any number of additional dimensions, including none
           weight: Tensor,  # (out_features,in_features)
           bias: Union[Tensor, None] = None,  # // (out_features) or ()
           ) -> Tensor:
    '''
    Applies a linear transformation to the incoming data: y = xA^T + b.
    y is (∗,out_features)
    '''

    input_shape = input.shape
    out_features, in_features = weight.shape
    assert in_features == input_shape[-1]

    N = math.prod(input_shape[0:-1])
    input_dims = len(input_shape)

    output_shape = (*input_shape[0:-1], out_features)
    if bias is None:
        y = Tensor(_infinicore.linear(input.view((N, in_features))._underlying,
                                      weight.permute((1, 0))._underlying))
    else:
        assert out_features == bias.shape[0]
        bias_shape = output_shape
        bias_strided = (0, 1)
        y = Tensor(_infinicore.linear_bias(input.view((N, in_features))._underlying,
                                           weight.permute((1, 0))._underlying,
                                           bias.as_strided(bias_shape, bias_strided)._underlying))
    return y.view(output_shape)


def linear_bk(input: Tensor,  # (∗,in_features) where * means any number of additional dimensions, including none
           weight: Tensor,  # (out_features,in_features)
           bias: Union[Tensor, None] = None,  # // (out_features) or ()
           ) -> Tensor:
    '''
    Applies a linear transformation to the incoming data: y = xA^T + b.
    y is (∗,out_features)
    '''

    input_shape = input.shape
    out_features, in_features = weight.shape
    assert in_features == input_shape[-1]

    N = math.prod(input_shape[0:-1])
    input_dims = len(input_shape)

    output_shape = (*input_shape[0:-1], out_features)
    if bias is None:
        y = Tensor(_infinicore.linear(input.view((N, in_features))._underlying,
                                      weight.permute((1, 0))._underlying))
    else:
        assert out_features == bias.shape[0]
        bias_shape = output_shape
        bias_strided = (0, 1)
        y = Tensor(_infinicore.linear_bias(input.view((N, in_features))._underlying,
                                           weight.permute((1, 0))._underlying,
                                           bias.as_strided(bias_shape, bias_strided)._underlying))
    return y.view(output_shape)