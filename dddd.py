import torch
import torch_musa 

a = torch.zeros((13, 4),dtype = torch.uint64)
b = torch.zeros((13, 4))
c = torch.stack([a, b]).sum(dim=0)

print(c)



#   self._torch_tensor = torch.randint(randint_low, randint_high, torch_shape, dtype=to_torch_dtype(dt), device=torch_device_map[device])