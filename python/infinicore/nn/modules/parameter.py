
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


from typing import Optional

import infinicore

class InfiniCoreParameter(infinicore.Tensor):
    def __init__(self, data: Optional[infinicore.Tensor] = None):
        self.data = data
        
Parameter = InfiniCoreParameter