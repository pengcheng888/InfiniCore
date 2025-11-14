import torch

import infinicore

# ------------------------------------------------
#  准备输入数据
# ------------------------------------------------

device_str = "cuda"
normalized_shape = 64
input_torch = torch.randn((1, 5, 64), device=device_str)
input_infini = infinicore.from_torch(input_torch)


# ------------------------------------------------
#  准备两种模型
# ------------------------------------------------
m_torch = torch.nn.RMSNorm(normalized_shape, device=device_str)
m_infini = infinicore.nn.RMSNorm(normalized_shape)

# ------------------------------------------------
#  准备两种参数
# ------------------------------------------------
param_torch_dict = {
    "weight": torch.rand(normalized_shape, device=device_str),
}

param_infini_dict = {
    "weight": infinicore.from_torch(param_torch_dict["weight"]),
}

# ------------------------------------------------
#  加载权重
# ------------------------------------------------
m_torch.load_state_dict(param_torch_dict)
m_infini.load_state_dict(param_infini_dict)


# ------------------------------------------------
#  计算
# ------------------------------------------------
out_torch = m_torch(input_torch)
out_infini = m_infini(input_infini)


# ------------------------------------------------
#  对比结果
# ------------------------------------------------
result = torch.zeros_like(out_torch)
ref = infinicore.from_blob(
    result.data_ptr(),
    out_infini.shape,
    dtype=out_infini.dtype,
    device=out_infini.device,
)

ref.copy_(out_infini)

# print(out_torch)
# print(result)

print("----------------------------------------")
print("abs error: ", torch.abs(out_torch - result).max())

if __name__ == "__main__":
    pass
