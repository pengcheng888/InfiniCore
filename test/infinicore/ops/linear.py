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
    print(torch.nn.functional.linear(a, b))


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


    class config:
        head_dim = 64
        max_position_embeddings = 10
        rope_theta = 10000.0

    a = infinicore.nn.RoPE(config())
    b = infinicore.nn.RoPE(config())


if __name__ == "__main__":
    func8_mul()
