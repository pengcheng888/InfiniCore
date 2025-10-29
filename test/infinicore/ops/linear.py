def func():
    import infinicore
    a = infinicore.empty((4, 4), dtype=infinicore.float32, device=infinicore.device("cuda", 0))
    b = infinicore.empty((4, 4), dtype=infinicore.float32, device=infinicore.device("cuda", 0))

    print(a)
    print(b)
    c = infinicore.linear(a, b, a)
    print(c)

    print("=============================================")

    print(a.shape)  # ok
    print(a.dtype)
    print(a.device)
    print(a.ndim)
    print(a.data_ptr)  # ok

    '''
    [3, 4]
    DataType.F32
    NVIDIA:0   缺少一个从 NVIDIA:0  变成 cuda:0 的函数  ???
    2
    <bound method Tensor.data_ptr of <infinicore.tensor.Tensor object at 0x7fd26e329900>>
    '''


def func7_mul():
    import infinicore
    from infinicore.nn.modules.linear import torch_tensor_2_infini_tensor
    import torch
    a = torch.randn((2, 2, 3))
    b = torch.randn((4, 3))
    bias = torch.randn((4))

    a_infini = torch_tensor_2_infini_tensor(a, device_str="cpu")
    b_infini = torch_tensor_2_infini_tensor(b, device_str="cpu")
    bias_infini = torch_tensor_2_infini_tensor(bias, device_str="cpu")

    print(a_infini)
    print(b_infini)
    y = infinicore.linear(a_infini, b_infini)
    print("============================>")
    print(y)
    print(torch.matmul(a, b.transpose(1, 0)))

    return

    a = torch.randn((2, 2, 3))
    b = torch.randn((3, 4))
    print(torch.matmul(a, b))

    a_shape = a.shape
    a = a.reshape(-1, 3)
    y = torch.matmul(a, b)
    print(y.reshape(2, 2, 4))


def func8_mul():
    import infinicore
    from infinicore.nn.modules.linear import torch_tensor_2_infini_tensor
    import torch
    a = torch.randn((2, 3))
    b = torch.randn((4, 3))
    bias = torch.randn((4))

    a_infini = torch_tensor_2_infini_tensor(a, device_str="cpu")
    b_infini = torch_tensor_2_infini_tensor(b, device_str="cpu")
    bias_infini = torch_tensor_2_infini_tensor(bias, device_str="cpu")

    print(a_infini)
    print(b_infini)
    y = infinicore.linear(a_infini, b_infini, bias_infini)
    print("============================>", y)

    print(torch.nn.functional.linear(a, b, bias))


def func9():
    import infinicore

    x = infinicore.empty((1, 2, 3), dtype=infinicore.float32, device=infinicore.device("cuda", 0))
    print(x.shape, x)

    y = x.permute((1, 0, 2))
    print(y.shape, y)

    print('=======================')
    import torch
    x = torch.randn((1, 2, 3))
    print(x.shape, x)
    y = x.permute((1, 0, 2))
    print(y.shape, y)


def func10():
    import infinicore
    from typing import Callable, Optional, Union
    class RoPE_infinicore_v2():
        sin_table: Union[infinicore.Tensor, None] = None
        cos_table: Union[infinicore.Tensor, None] = None

        def __init__(self, config):
            self.max_position_embeddings = config.max_position_embeddings
            self.rope_theta = config.rope_theta
            self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

            if self.get_sin_table() is None:
                print("create")

                sin_table, cos_table = self.create_sin_cos_table(self.max_position_embeddings, head_dim=self.head_dim, theta=self.rope_theta)
                print(sin_table.shape)
                print(cos_table.shape)

                from infinicore.nn.modules.linear import create_infinicore_tensor, infini_tensor_2_torch_tensor
                device_str = "cpu"
                RoPE_infinicore_v2.sin_table = create_infinicore_tensor(sin_table, device_str)
                RoPE_infinicore_v2.cos_table = create_infinicore_tensor(cos_table, device_str)

            print(id(RoPE_infinicore_v2.sin_table))

            print(id(self.get_sin_table()))

        def get_sin_table(self):
            return RoPE_infinicore_v2.sin_table

        def get_cos_table(self):
            return RoPE_infinicore_v2.cos_table

        def create_sin_cos_table(self, max_position, head_dim=64, theta=10000.0):
            import torch

            assert head_dim % 2 == 0, "Embedding dimension must be even."
            pos = torch.arange(0, max_position)
            freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
            angles = torch.outer(pos, freqs)
            return torch.sin(angles), torch.cos(angles)

    class config:
        head_dim = 64
        max_position_embeddings = 10
        rope_theta = 10000.0

    a = RoPE_infinicore_v2(config())
    b = RoPE_infinicore_v2(config())
    b = RoPE_infinicore_v2(config())


if __name__ == "__main__":
    # func8_mul()
    func10()
    exit()


    def create_sin_cos_table(max_position, dim=64, theta=10000.0):
        import torch
        pos = torch.range(0, max_position)
        print(pos)
        assert dim % 2 == 0, "Embedding dimension must be even."
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        angles = torch.outer(pos, freqs)
        return torch.sin(angles), torch.cos(angles)


    sin_table, cos_table = create_sin_cos_table(10)
    print(sin_table.shape)
    print(cos_table.shape)
