# Copyright (c) 2025, InfiniCore
#
# This file contains modified code derived from PyTorch's `torch.nn.Module`
# implementation, which is licensed under the BSD 3-Clause License.
#
# The modifications include adaptations for the InfiniCore framework, custom
# parameter/buffer registration mechanisms, and simplified state_dict handling.
#
# Original PyTorch source:
# https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py
#
# Referencing PyTorch v2.4.0
#
# The use of this file is governed by the BSD 3-Clause License.

from collections import OrderedDict, namedtuple
from typing import Any, Dict, Iterator, List, Mapping, Optional, Set, Tuple, Union

import infinicore

from .parameter import Parameter


class _IncompatibleKeys(
    namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])
):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super().__repr__()

    __str__ = __repr__


class InfiniCoreModule:
    r"""Base class for InfiniCore neural network modules.
    Your models should also subclass this class.
    Modules can also contain other Modules, allowing to nest them in a tree structure.
    """

    _version: int = 1
    _parameters: Dict[str, Optional[Parameter]]
    _modules: Dict[str, Optional["InfiniCoreModule"]]

    def __init__(self):
        super().__setattr__("_parameters", OrderedDict())
        super().__setattr__("_modules", OrderedDict())

    def __getattr__(self, name: str) -> Any:
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]

        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Union[infinicore.Tensor, "Module"]) -> None:
        def remove_from(*dicts_or_sets) -> None:
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get("_parameters")
        if params is None:
            raise AttributeError(
                "cannot assign parameters before Module.__init__() call"
            )

        if isinstance(value, Parameter):  # the value is of type Parameter
            remove_from(self.__dict__, self._modules)
            self.register_parameter(name, value)
        elif name in params:  # value will overwrite the name of params.
            if not isinstance(value, infinicore.Tensor):
                raise TypeError(
                    f"cannot assign 'value' as parameter '{name}'  (infinicore.nn.Parameter, Parameter or None expected)"
                )
            self.register_parameter(name, value)

        else:
            modules = self.__dict__.get("_modules")
            if modules is None:
                raise AttributeError(
                    "cannot assign module before Module.__init__() call"
                )

            if isinstance(value, InfiniCoreModule):
                remove_from(self.__dict__, self._modules)
                modules[name] = value
            elif name in modules:  # Do not overwrite this variable
                raise TypeError(
                    f"cannot assign 'value' as child module '{name}' (infinicore.nn.Module or None expected)"
                )
            else:
                super().__setattr__(name, value)

    def register_parameter(self, name: str, param: Parameter) -> None:
        if "_parameters" not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call"
            )
        elif not isinstance(name, str):
            raise TypeError("parameter name should be a string.")
        elif "." in name:
            raise KeyError('parameter name can\'t contain "."')
        elif name == "":
            raise KeyError('parameter name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"attribute '{name}' already exists")

        if param is None:
            self._parameters[name] = None  # 竟然可以是None
        else:
            if not isinstance(param, (Parameter, infinicore.Tensor)):
                raise TypeError(
                    f"cannot assign  'param' object to parameter '{name}' "
                    "(infinicore.nn.Parameter, Parameter or None required)"
                )

            self._parameters[name] = param
            super().__setattr__(name, param)

    def add_module(self, name: str, module: Optional["Module"]) -> None:
        r"""Add a child module to the current module.
        The module can be accessed as an attribute using the given name.
        """
        if not isinstance(name, str):
            raise TypeError(f"module name should be a string. Got {name}")
        elif "." in name:
            raise KeyError(f'module name can\'t contain ".", got: {name}')
        elif name == "":
            raise KeyError('module name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute '{name}' already exists")

        if module is not None and not isinstance(module, Module):
            raise TypeError(f"{module} is not a Module subclass")

        self._modules[name] = module

    def register_module(self, name: str, module: Optional["Module"]) -> None:
        r"""Alias for :func:`add_module`."""
        self.add_module(name, module)

    def get_extra_state(self) -> Any:
        """Return any extra state to include in the module's state_dict."""
        raise RuntimeError(
            "Reached a code path in Module.get_extra_state() that should never be called. "
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        local_name_params = self._parameters.items()
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]

                # input_param must be of type infinicore.Tensor
                if not isinstance(input_param, infinicore.Tensor):
                    raise TypeError(
                        f"While copying the parameter named {key}, expected infinicore.Tensor from checkpoint but received {type(input_param)}"
                    )

                if (
                    (param.shape == input_param.shape)
                    and (param.dtype == input_param.dtype)
                    and (param.device == input_param.device)
                ):
                    param.copy_(input_param)
                else:
                    print(f"param '{name}' don't match input_param '{key}'")
                    setattr(self, name, input_param)

            elif strict:
                missing_keys.append(key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix):
                    input_name = key[len(prefix) :].split(".", 1)
                    # Must be Module if it have attributes
                    if len(input_name) > 1:
                        if input_name[0] not in self._modules:
                            unexpected_keys.append(key)
                    elif input_name[0] not in local_state:
                        unexpected_keys.append(key)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                "Expected state_dict to be dict-like, got {}.".format(type(state_dict))
            )

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = OrderedDict(state_dict)
        if metadata is not None:
            state_dict._metadata = metadata  # type: ignore[attr-defined]

        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        def load(module, local_state_dict, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                local_state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    child_prefix = prefix + name + "."
                    child_state_dict = {
                        k: v
                        for k, v in local_state_dict.items()
                        if k.startswith(child_prefix)
                    }
                    load(child, child_state_dict, child_prefix)

        load(self, state_dict)
        del load

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0,
                    "Unexpected key(s) in state_dict: {}. ".format(
                        ", ".join('"{}"'.format(k) for k in unexpected_keys)
                    ),
                )
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0,
                    "Missing key(s) in state_dict: {}. ".format(
                        ", ".join('"{}"'.format(k) for k in missing_keys)
                    ),
                )

        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    self.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    def parameters(self, recurse: bool = True) -> Iterator["Parameter"]:
        r"""Returns an iterator over module parameters.

        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            Parameter: module parameter

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for param in model.parameters():
            ...     print(type(param), param.size())

        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, "Parameter"]]:
        r"""Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself.

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            (str, Parameter): Tuple containing the name and parameter

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, param in self.named_parameters():
            ...     if name in ['bias']:
            ...         print(param.size())

        """
        gen = self._named_members(
            lambda module: module._parameters.items(), prefix=prefix, recurse=recurse
        )
        for elem in gen:
            yield elem

    def modules(self) -> Iterator["Module"]:
        r"""Returns an iterator over all modules in the network.

        Yields:
            Module: a module in the network

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.modules()):
            ...     print(idx, '->', m)

            0 -> Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
            1 -> Linear(in_features=2, out_features=2, bias=True)

        """
        for name, module in self.named_modules():
            yield module

    def named_modules(
        self,
        memo: Optional[Set["InfiniCoreModule"]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        r"""Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.

        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances in the result
                or not

        Yields:
            (str, Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.named_modules()):
            ...     print(idx, '->', m)

            0 -> ('', Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            ))
            1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

        """
        if memo is None:
            memo = set()
        if remove_duplicate:
            if self in memo:
                return
            memo.add(self)
        yield prefix, self
        for name, module in self._modules.items():
            if module is None:
                continue
            submodule_prefix = prefix + ("." if prefix else "") + name
            # Handle both InfiniCoreModule and torch.nn.Module
            if isinstance(module, InfiniCoreModule):
                for m in module.named_modules(memo, submodule_prefix, remove_duplicate):
                    yield m
            elif isinstance(module, infinicore.nn.Module):
                # For torch.nn.Module, use its named_modules method
                # torch.nn.Module.named_modules returns (name, module) tuples
                for sub_name, sub_module in module.named_modules(
                    prefix=submodule_prefix, remove_duplicate=remove_duplicate
                ):
                    yield (sub_name, sub_module)

    def children(self) -> Iterator["Module"]:
        r"""Returns an iterator over immediate children modules.

        Yields:
            Module: a child module (can be InfiniCoreModule or infinicore.nn.Module)
        """
        for name, module in self.named_children():
            yield module

    def named_children(
        self,
    ) -> Iterator[Tuple[str, "Module"]]:
        r"""Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself.

        Yields:
            (str, Module): Tuple containing a name and child module

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, module in model.named_children():
            >>>     if name in ['conv4', 'conv5']:
            >>>         print(module)

        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def to(self, *args, **kwargs):
        raise ValueError("not supported!")


Module = InfiniCoreModule
