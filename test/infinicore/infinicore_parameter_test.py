import safetensors.torch
import torch
import torch.nn as nn
import safetensors

# ============================================================
# 0. infinicore 包导入，配置测试用 safetensors 临时存储路径
# ============================================================
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python/infinicore')))

save_dir = os.path.join(os.path.dirname(__file__), '../../tmp')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "infinicore_parameter_test.safetensors")

# ============================================================
# 1. 使用 PyTorch 定义并保存模型（使用 torch.nn.Parameter）
# ============================================================

class TorchParameterNet(nn.Module):
    def __init__(self, in_features=10, out_features=5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.scale = nn.Parameter(torch.ones(1) * 0.5)
        self.register_buffer("offset", torch.tensor(0.1))

    def forward(self, x):
        return (x @ self.weight.t() + self.bias) * self.scale + self.offset


# ===== 保存 Torch 模型 =====
torch_model = TorchParameterNet()
torch_state_dict = torch_model.state_dict()
safetensors.torch.save_file(torch_state_dict, save_path)
print("✓ PyTorch 模型已保存")

# ============================================================
# 2. 使用 torch 方式加载并推理
# ============================================================

torch_model_infer = TorchParameterNet()
torch_model_infer.load_state_dict(safetensors.torch.load_file(save_path))
torch_model_infer.eval()

input = torch.randn(2, 10)
torch_model_out = torch_model_infer(input)
print("✓ Torch 输出：", torch_model_out.detach().numpy().mean())

# ============================================================
# 3. 使用 Parameter 加载并推理
# ============================================================

from nn.modules import Module, Parameter

class InfiniCoreParameterNet(Module):
    def __init__(self, in_features=10, out_features=5):
        super().__init__()
        # 使用 Parameter 替代 torch.nn.Parameter
        self.weight = Parameter(torch.randn(out_features, in_features))
        self.bias = Parameter(torch.randn(out_features))
        self.scale = Parameter(torch.ones(1) * 0.5)
        self.register_buffer("offset", torch.tensor(0.1))

    def forward(self, x):
        return (x @ self.weight.t() + self.bias) * self.scale + self.offset

# ===== 使用 InfiniCoreParameterNet 读取 safetensors 并推理 =====
infinicore_model_infer = InfiniCoreParameterNet()
infinicore_model_infer.load_state_dict(safetensors.torch.load_file(save_path))
infinicore_model_infer.eval()

infinicore_model_out = infinicore_model_infer.forward(input)
print("✓ InfiniCore 输出：", infinicore_model_out.detach().numpy().mean())

# ============================================================
# 4. 对比结果
# ============================================================

diff = (infinicore_model_out - torch_model_out).abs().max().item()
print(f"✓ Parameter 与 Torch 最大误差: {diff:.8f}")
if diff < 1e-9:
    print("✓ Parameter 与 Torch 精度一致.")
else:
    print("✗ Parameter 与 Torch 精度存在差异.")

# ============================================================
# 5. 测试 Parameter 的基本功能
# ============================================================

print("\n=== 测试 Parameter 基本功能 ===")

# 测试 1: 创建 Parameter
param1 = Parameter(torch.randn(5, 10))
print(f"✓ 创建 Parameter，形状: {param1.shape}")
# 检查是否是 Parameter 类型（可能是 InfiniCoreParameter 的别名）
from nn.modules.parameter import InfiniCoreParameter
assert isinstance(param1, (Parameter, InfiniCoreParameter)), "应该是 Parameter 类型"
assert isinstance(param1, torch.Tensor), "应该是 torch.Tensor 的子类"

# 测试 2: requires_grad
param2 = Parameter(torch.randn(3, 4), requires_grad=False)
print(f"✓ 创建 requires_grad=False 的 Parameter: {param2.requires_grad}")
assert not param2.requires_grad, "requires_grad 应该为 False"

param3 = Parameter(torch.randn(3, 4), requires_grad=True)
print(f"✓ 创建 requires_grad=True 的 Parameter: {param3.requires_grad}")
assert param3.requires_grad, "requires_grad 应该为 True"

# 测试 3: 自动注册到 Module
class TestModule(Module):
    def __init__(self):
        super().__init__()
        self.weight = Parameter(torch.randn(5, 10))
        self.bias = Parameter(torch.randn(5))

test_module = TestModule()
param_count = sum(1 for _ in test_module.parameters())
print(f"✓ 自动注册到 Module，参数数量: {param_count}")
assert param_count == 2, f"应该有 2 个参数，实际为 {param_count}"

# 测试 4: 参数访问
assert test_module.weight is not None, "weight 应该可以访问"
assert test_module.bias is not None, "bias 应该可以访问"
print("✓ 参数可以通过属性访问")

# 测试 5: state_dict
state_dict = test_module.state_dict()
print(f"✓ state_dict 键数量: {len(state_dict)}")
assert 'weight' in state_dict, "state_dict 应该包含 weight"
assert 'bias' in state_dict, "state_dict 应该包含 bias"
print(f"✓ state_dict 键: {list(state_dict.keys())}")

# 测试 6: __repr__
repr_str = repr(param1)
print(f"✓ __repr__ 方法: 输出包含类名")
assert "Parameter" in repr_str or "InfiniCoreParameter" in repr_str, "repr 应该包含类名"
print(repr_str[:100] + "...")

# 测试 7: 与 torch.nn.Parameter 兼容性
class MixedModule(Module):
    def __init__(self):
        super().__init__()
        self.torch_param = nn.Parameter(torch.randn(3, 4))
        self.infinicore_param = Parameter(torch.randn(3, 4))

mixed_module = MixedModule()
mixed_param_count = sum(1 for _ in mixed_module.parameters())
print(f"✓ 混合使用 torch.nn.Parameter 和 Parameter，参数数量: {mixed_param_count}")
assert mixed_param_count == 2, f"应该有 2 个参数，实际为 {mixed_param_count}"

# 测试 8: 前向传播
class TestModuleWithForward(Module):
    def __init__(self):
        super().__init__()
        self.weight = Parameter(torch.randn(5, 10))
        self.bias = Parameter(torch.randn(5))
    
    def forward(self, x):
        return x @ self.weight.t() + self.bias

test_module_forward = TestModuleWithForward()
test_input = torch.randn(2, 10)
with torch.no_grad():
    output = test_module_forward.forward(test_input)
print(f"✓ 前向传播成功，输出形状: {output.shape}")
assert output.shape == (2, 5), f"输出形状应该是 (2, 5)，实际为 {output.shape}"

# 测试 9: 从 None 创建
param_empty = Parameter(None)
print(f"✓ 从 None 创建 Parameter，形状: {param_empty.shape}")
assert param_empty.shape == torch.Size([0]), "从 None 创建应该是空张量"

# 测试 10: 深拷贝
import copy
param_copy = copy.deepcopy(param1)
print(f"✓ 深拷贝 Parameter，形状: {param_copy.shape}")
assert param_copy.shape == param1.shape, "深拷贝后形状应该相同"
assert not torch.equal(param_copy, param1) or id(param_copy) != id(param1), "深拷贝应该是新对象"

print("\n=== 所有测试通过! ===")

