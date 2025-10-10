def func1():
    import torch
    from torch import nn

    m = nn.Linear(20, 30)
    input = torch.randn(128, 20)
    output = m(input)
    print(output.size())


def func2():
    import infinicore
    from infinicore import nn
    import torch

    m = nn.Linear(20, 30)
    input = torch.randn(128, 20, dtype=torch.float32)
    output = m.forward_torch(input)
    print(output)
    print('=============================')
    device_str = "cpu"
    infini_c = infinicore.empty(
        (2, 3), dtype=infinicore.float32, device=infinicore.device(device_str, 0)
    )

    # -----------------------------------
    from infinicore.nn.modules.linear import create_infinicore_tensor, print_infini_tensor

    input_infini = create_infinicore_tensor(input, device_str)

    output = m.forward(input_infini)
    print_infini_tensor(output)


def func3():
    import infinicore
    from infinicore import nn
    import torch

    m = nn.Linear(5, 5)
    input = torch.randn(5, 5, dtype=torch.float32)
    output = m.forward_torch(input)
    # print(m.weight)
    # print(input)
    print(output)

    print('=============================')
    # -----------------------------------
    from infinicore.nn.modules.linear import create_infinicore_tensor, print_infini_tensor
    device_str = "cpu"
    input_infini = create_infinicore_tensor(input, device_str)

    output = m.forward(input_infini)
    # print_infini_tensor(m.weight_infini)
    # print_infini_tensor(input_infini)
    print_infini_tensor(output)


def func4():
    import infinicore
    infini_c = infinicore.empty((3, 4), dtype=infinicore.float32, device=infinicore.device("cuda", 0))

    print(infini_c.shape)  # ok
    print(infini_c.dtype)
    print(infini_c.device)
    print(infini_c.ndim)
    print(infini_c.data_ptr)  # ok

    '''
    [3, 4]
    DataType.F32
    NVIDIA:0   缺少一个从 NVIDIA:0  变成 cuda:0 的函数  ???
    2
    <bound method Tensor.data_ptr of <infinicore.tensor.Tensor object at 0x7fd26e329900>>
    '''


if __name__ == '__main__':
    func4()
