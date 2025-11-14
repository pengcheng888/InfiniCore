import infinicore
import torch

# bnsd
weight = torch.rand(1, 4, 10, 3, dtype=torch.float32, device="cpu").to_infini()


w1 = infinicore.narrow(weight, 2, 0, 1)
w2 = infinicore.narrow(weight, 2, 1, 1)
print(w1)
print(w2)
print(w1.data_ptr(), w2.data_ptr())

print("=====================================")
w1.copy_(w2)
print(w1.data_ptr(), w2.data_ptr())

# y1 = torch.rand(1, 32, 2, 64, dtype=torch.float32, device="cpu").to_infini()
# y2 = torch.rand(1, 32, 2, 64, dtype=torch.float32, device="cpu").to_infini()
# print(y1 + y2)
