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

from collections import OrderedDict
from typing import  Any, Optional, Union, Mapping, Dict, List
from .parameter import Parameter

import infinicore

class InfiniCoreModule:
    r"""Base class for InfiniCore neural network modules.
    Your models should also subclass this class.

    Modules can also contain other Modules, allowing 
    to nest them in a tree structure.
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
            
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Union[infinicore.Tensor, 'InfiniCoreModule']) -> None:
        def remove_from(*dicts_or_sets) -> None:
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get("_parameters")
        if params is None:
            raise AttributeError("cannot assign parameters before Module.__init__() call")
    
        if isinstance(value, Parameter):  # value是Parameter
            remove_from(self.__dict__,self._modules )
            self.register_parameter(name, value)
        elif name in params: # value不是Parameter，想要覆盖 name
            if not isinstance(value, infinicore.Tensor):
                raise TypeError( f"cannot assign 'value' as parameter '{name}'  (infinicore.nn.Parameter, Parameter or None expected)")
            self.register_parameter(name, value)

        else:
            modules = self.__dict__.get("_modules")
            if modules is None:
                raise AttributeError("cannot assign module before Module.__init__() call")
            
            if isinstance(value, InfiniCoreModule):
                remove_from(self.__dict__,self._modules )
                modules[name] = value # 为啥不是注册呢？？
            elif name in modules:
                # 不让覆盖
                raise TypeError( f"cannot assign 'value' as child module '{name}' (infinicore.nn.Module or None expected)" )
            else:
                # 先不判断buffer
                # 直接赋值
                super().__setattr__(name, value)
 
    def add_module(self, name: str, module: Optional["Module"]) -> None:
        r"""Add a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (str): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module or None): child module to be added to the module. If
                ``None``, then operations that run on modules, such as :attr:`eval`,
                are ignored. If ``None``, the module is **not** included in the
                module's :attr:`children`.
        """
        if not isinstance(name, str):
            raise TypeError(f"module name should be a string. Got {name}")
        elif '.' in name:
            raise KeyError(f"module name can't contain \".\", got: {name}")
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute '{name}' already exists")
        
        if module is not None and not isinstance(module, infinicore.nn.Module):
            raise TypeError(f"{module} is not a Module subclass")
        
        self._modules[name] = module

    def register_parameter(self, name: str, param:Parameter) -> None:
        
        if "_parameters" not in self.__dict__:
            raise AttributeError(  "cannot assign parameter before Module.__init__() call"  )
        elif not isinstance(name, str):
            raise TypeError( "parameter name should be a string."  )
        elif "." in name:
            raise KeyError('parameter name can\'t contain "."')
        elif name == "":
            raise KeyError('parameter name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"attribute '{name}' already exists")
        
        if param is None:
            self._parameters[name] = None # 竟然可以是None
        else:
            if not isinstance(param, (Parameter,infinicore.Tensor)):
                raise TypeError(
                    f"cannot assign  'param' object to parameter '{name}' "
                    "(infinicore.nn.Parameter, Parameter or None required)"
                )

            self._parameters[name] = param
            super().__setattr__(name, param)
            
            
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        local_name_params = self._parameters.items()
        local_state = {k: v for k, v in local_name_params if v is not None}
        print("_load_from_state_dict:   ",local_state)

        for name, param in local_state.items():
            key = prefix + name
            print(key)
            if key in state_dict:
                input_param = state_dict[key]
                
                if not isinstance(input_param, infinicore.Tensor): # state_dict 中要都是 infinicore.Tensor 类型
                    raise TypeError(f'While copying the parameter named {key}, expected infinicore.Tensor from checkpoint but received {type(input_param)}')

               
                # ------------ 直接替换，是浅拷贝，不是复制。
                # print(f"param  {name}  {key}  {type(param) }     {type(param.data) }       {type(input_param)}"  ,param, input_param.shape)
                # param = input_param
                # setattr(param, input_param)
                # self._parameters[name] = input_param # 把Parameter覆盖，变为 infinicore.Tensor
                setattr(self, name, input_param)

            elif strict:
                missing_keys.append(key)

        
        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix) :
                    input_name = key[len(prefix):].split(".", 1)
                    # Must be Module if it have attributes
                    if len(input_name) > 1:
                        if input_name[0] not in self._modules:
                            unexpected_keys.append(key)
                    elif input_name[0] not in local_state:
                        unexpected_keys.append(key)

        
    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True):
        if not isinstance(state_dict, Mapping):
            raise TypeError("Expected state_dict to be dict-like, got {}.".format(type(state_dict)))

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = OrderedDict(state_dict)
        if metadata is not None:
            state_dict._metadata = metadata  # type: ignore[attr-defined]
        
        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        def load(module, local_state_dict, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                local_state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    child_prefix = prefix + name + '.'
                    child_state_dict = {k: v for k, v in local_state_dict.items() if k.startswith(child_prefix)}
                    load(child, child_state_dict, child_prefix)

        load(self, state_dict)
        del load


Module =  InfiniCoreModule
