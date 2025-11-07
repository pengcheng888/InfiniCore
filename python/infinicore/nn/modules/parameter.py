# Copyright (c) 2025, InfiniCore
# 
# This file contains modified code derived from PyTorch's `torch.nn.Parameter`
# implementation, which is licensed under the BSD 3-Clause License.
#
# The modifications include adaptations for the InfiniCore framework.
#
# Original PyTorch source:
# https://github.com/pytorch/pytorch/blob/main/torch/nn/parameter.py
#
# Referencing PyTorch v2.4.0
#
# The use of this file is governed by the BSD 3-Clause License.

import torch
from typing import Optional
from collections import OrderedDict


class InfiniCoreParameter(torch.Tensor):
    r"""A kind of Tensor that is to be considered a module parameter.

    Parameters are :class:`~torch.Tensor` subclasses, that have a
    very special property when used with :class:`InfiniCoreModule` s - when they're
    assigned as Module attributes they are automatically added to the list of
    its parameters, and will appear e.g. in :meth:`~InfiniCoreModule.parameters` iterator.

    Assigning a Tensor doesn't have such effect. This is because one might
    want to cache some temporary state, like last hidden state of the RNN, in
    the model. If there was no such class as :class:`InfiniCoreParameter`, these
    temporaries would get registered too.

    Args:
        data (Tensor, optional): parameter tensor. If None, creates an empty tensor.
        requires_grad (bool, optional): if the parameter requires gradient. Note that
            the torch.no_grad() context does NOT affect the default behavior of
            Parameter creation--the Parameter will still have `requires_grad=True` in
            :class:`~no_grad` mode. See :ref:`locally-disable-grad-doc` for more
            details. Default: `True`

    Example::

        >>> import torch
        >>> from infinicore.nn.modules import InfiniCoreModule, InfiniCoreParameter
        >>> 
        >>> class MyModule(InfiniCoreModule):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.weight = InfiniCoreParameter(torch.randn(10, 5))
        ...         self.bias = InfiniCoreParameter(torch.randn(5))
        ...
        >>> module = MyModule()
        >>> for param in module.parameters():
        ...     print(param.shape)
        torch.Size([10, 5])
        torch.Size([5])
    """

    def __new__(cls, data: Optional[torch.Tensor] = None, requires_grad: bool = True):
        if data is None:
            data = torch.empty(0)
        
        # Handle standard torch.Tensor or InfiniCoreParameter
        if type(data) is torch.Tensor or type(data) is InfiniCoreParameter:
            # For ease of BC maintenance, keep this path for standard Tensor.
            # Eventually (tm), we should change the behavior for standard Tensor to match.
            return torch.Tensor._make_subclass(cls, data, requires_grad)
        
        # Path for custom tensors: set a flag on the instance to indicate parameter-ness.
        t = data.detach().requires_grad_(requires_grad)
        
        if type(t) is not type(data):
            raise RuntimeError(
                f"Creating a InfiniCoreParameter from an instance of type {type(data).__name__} "
                "requires that detach() returns an instance of the same type, but return "
                f"type {type(t).__name__} was found instead. To use the type as a "
                "InfiniCoreParameter, please correct the detach() semantics defined by "
                "its __torch_dispatch__() implementation."
            )
        
        t._is_param = True
        return t

    # Note: the 3 methods below only apply to standard Tensor. Parameters of custom tensor types
    # are still considered that custom tensor type and these methods will not be called for them.

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(
                self.data.clone(memory_format=torch.preserve_format), self.requires_grad
            )
            memo[id(self)] = result
            return result

    def __repr__(self):
        return "InfiniCoreParameter containing:\n" + super().__repr__()

    def __reduce_ex__(self, proto):
        # Simplified version for serialization
        # In a full implementation, you might want to handle hooks and state
        state = getattr(self, '_state', None)
        hooks = OrderedDict()
        
        if not state:
            return (
                _rebuild_parameter,
                (self.data, self.requires_grad, hooks),
            )
        return (
            _rebuild_parameter_with_state,
            (self.data, self.requires_grad, hooks, state),
        )

    # Note: __torch_function__ is handled by the Tensor base class
    # We don't need to override it for standard Parameter behavior


def _rebuild_parameter(data, requires_grad, hooks):
    """Rebuild a parameter from serialized data."""
    param = InfiniCoreParameter(data, requires_grad)
    # Apply hooks if any (simplified - full implementation would restore hooks)
    return param


def _rebuild_parameter_with_state(data, requires_grad, hooks, state):
    """Rebuild a parameter with extra state from serialized data."""
    param = InfiniCoreParameter(data, requires_grad)
    param._state = state
    # Apply hooks if any (simplified - full implementation would restore hooks)
    return param

