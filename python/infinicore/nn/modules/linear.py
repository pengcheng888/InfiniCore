import torch
import infinicore
from typing import Union


def to_infinicore_dtype(torch_dtype):
    """Convert PyTorch data type to infinicore data type"""
    
    if torch_dtype == torch.float32:
        return infinicore.float32
    elif torch_dtype == torch.float16:
        return infinicore.float16
    elif torch_dtype == torch.bfloat16:
        return infinicore.bfloat16
    elif torch_dtype == torch.int32:
        return infinicore.int32
    elif torch_dtype == torch.int64:
        return infinicore.int64
    else:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")

def to_torch_dtype(infini_dtype):
    """Convert infinicore data type to PyTorch data type"""
    if infini_dtype == infinicore.float16:
        return torch.float16
    elif infini_dtype == infinicore.float32:
        return torch.float32
    elif infini_dtype == infinicore.bfloat16:
        return torch.bfloat16
    elif infini_dtype == infinicore.int32:
        return torch.int32
    elif infini_dtype == infinicore.int64:
        return torch.int64
    else:
        raise ValueError(f"Unsupported infinicore dtype: {infini_dtype}")


def create_infinicore_tensor(torch_tensor: torch.Tensor, device_str):
    """Create infinicore tensor from PyTorch tensor"""
    infini_temp = infinicore.from_blob(
                torch_tensor.data_ptr(),
                list(torch_tensor.shape),
                dtype=to_infinicore_dtype(torch_tensor.dtype),
                device=infinicore.device(device_str, 0),
            )

    infini_tensor = infinicore.empty( torch_tensor.shape,
                                 dtype= to_infinicore_dtype(torch_tensor.dtype), 
                                 device=infinicore.device(device_str, 0)
    )
    infini_tensor.copy_(infini_temp)
    return infini_tensor


def torch_tensor_2_infini_tensor(torch_tensor: torch.Tensor, device_str="cpu"):
    """Create infinicore tensor from PyTorch tensor"""

    return create_infinicore_tensor(torch_tensor, device_str)
    print("--------------->")
    print(torch_tensor)
    infini_temp = infinicore.from_blob(
                torch_tensor.data_ptr(),
                list(torch_tensor.shape),
                dtype=to_infinicore_dtype(torch_tensor.dtype),
                device=infinicore.device(device_str, 0),
            )
    print_infini_tensor(infini_temp)
    return infini_temp
    print("infini_temp",infini_temp)
    infini_tensor = infinicore.empty( torch_tensor.shape,
                                 dtype= to_infinicore_dtype(torch_tensor.dtype), 
                                 device=infinicore.device(device_str, 0)
    )
    infini_tensor.copy_(infini_temp)
    return infini_tensor

def infini_tensor_2_torch_tensor(infini_tensor:infinicore.Tensor, device_str="cpu"):
    torch_tensor = torch.rand(infini_tensor.shape, dtype= to_torch_dtype(infini_tensor.dtype) , device=device_str)

    temp_tensor =  infinicore.from_blob(
            torch_tensor.data_ptr(),
            list(torch_tensor.shape),
            dtype=to_infinicore_dtype(torch_tensor.dtype),
            device=infinicore.device(device_str, 0),
        )
    temp_tensor.copy_(infini_tensor)
    return torch_tensor

def print_infini_tensor(infini_tensor, device_str="cpu"):
    torch_tensor = torch.rand(infini_tensor.shape, dtype= to_torch_dtype(infini_tensor.dtype) , device=device_str)

    # temp_tensor = create_infinicore_tensor(torch_tensor, device_str)
    temp_tensor =  infinicore.from_blob(
            torch_tensor.data_ptr(),
            list(torch_tensor.shape),
            dtype=to_infinicore_dtype(torch_tensor.dtype),
            device=infinicore.device(device_str, 0),
        )
    temp_tensor.copy_(infini_tensor)



class Linear(torch.nn.Linear):
    shared_count = 0
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = False,
                 device=None,
                 dtype=None) -> None:
        super().__init__(in_features=in_features,
                         out_features=out_features,
                         bias=False,
                         device=device,
                         dtype=dtype)
        pass

    def forward_torch(self, input: torch.Tensor) -> torch.Tensor:
        # print('forward_torch')

        # in_features 是 20
        # out_features 是 30
        # 将输入的维度从 in_features 变成 out_features
        # F.linear(input, self.weight, self.bias) # size([128,in_features]),size([out_features,in_features]) ==> size([128,out_features])
        return super().forward(input)

    def forward_infinicore(self, input: infinicore.Tensor) -> infinicore.Tensor:
        device_str = "cpu"
        self.weight_transpose = self.weight.T.contiguous()
        self.weight_infini = create_infinicore_tensor(self.weight_transpose, device_str)

        Linear.shared_count += 1
        #print('Linear forward_infinicore ', Linear.shared_count)
        # size([128,in_features]), size([in_features,out_features]) ==> size([128,out_features])
        return infinicore.matmul(input, self.weight_infini)

    def forward(self,
                input: Union[infinicore.Tensor, torch.Tensor],
                use_infinicore: bool = False
                ) -> Union[infinicore.Tensor, torch.Tensor]:
        if isinstance(input, torch.Tensor):
            # ----------------------------------------------- #
            # 输入是 torch.Tensor，使用 torch 算子计算 
            # ----------------------------------------------- #
            if not use_infinicore:
                return self.forward_torch(input)
            
            # ----------------------------------------------- #
            # 输入是 torch.Tensor，使用 infinicore 算子计算 
            # ----------------------------------------------- #
            input_shape = input.shape
            input_torch = input.reshape((-1, input_shape[-1]))
            device_str = "cpu"
            input_infinicore = create_infinicore_tensor(input_torch, device_str)
            output_infinicore = self.forward_infinicore(input_infinicore)
            output_torch = infini_tensor_2_torch_tensor(output_infinicore)

            output =  output_torch.reshape((input_shape[0],input_shape[1],-1))
            return output
        # ----------------------------------------------- #
        # 输入是 infinicore.Tensor，使用 infinicore 算子计算  
        # ----------------------------------------------- #
        return self.forward_infinicore(input)

    def extra_repr(self) -> str:
        return f" infinicore op : in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

    def testOP(self):
        pass