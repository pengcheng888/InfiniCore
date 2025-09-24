import torch

a = torch.rand((13, 4))
b = torch.rand((13, 4))



c = torch.stack([a, b]).sum(dim=0)

print(c)
