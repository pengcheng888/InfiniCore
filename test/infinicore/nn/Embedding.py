import torch

import infinicore

# ------------------------------------------------
#  准备输入数据
# ------------------------------------------------

device_str = "cpu"
num_embeddings = 10
embedding_dim = 3
input_torch = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]]).to(device=device_str)
input_infini = infinicore.from_torch(input_torch.to(device="cpu"))


# ------------------------------------------------
#  准备两种模型
# ------------------------------------------------
m_torch = torch.nn.Embedding(num_embeddings, embedding_dim, device=device_str)
m_infini = infinicore.nn.Embedding(num_embeddings, embedding_dim)

# ------------------------------------------------
#  准备两种参数
# ------------------------------------------------
param_torch_dict = {
    "weight": torch.rand((num_embeddings, embedding_dim), device=device_str),
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
