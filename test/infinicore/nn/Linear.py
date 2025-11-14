import torch
import infinicore


# ------------------------------------------------
#  准备输入数据
# ------------------------------------------------

device_str = "cuda"

out_features, in_features = 20, 30

input_torch = torch.randn(128, in_features, device=device_str)
input_infini = infinicore.from_torch(input_torch)


# ------------------------------------------------
#  准备两种模型
# ------------------------------------------------
m_torch = torch.nn.Linear(in_features, out_features, bias=True, device=device_str)
m_infini = infinicore.nn.Linear(in_features, in_features, bias=True)

# ------------------------------------------------
#  准备两种参数
# ------------------------------------------------
param_torch_dict = {
    "weight": torch.rand(out_features, in_features, device=device_str),
    "bias": torch.rand(out_features, device=device_str),
}

param_infini_dict = {
    "weight": infinicore.from_torch(param_torch_dict["weight"]),
    "bias": infinicore.from_torch(param_torch_dict["bias"]),
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
