import torch


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
    from infinicore.nn.modules.linear import create_infinicore_tensor

    input_infini = create_infinicore_tensor(input, device_str)

    output = m.forward(input_infini)
    print(output)


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
    from infinicore.nn.modules.linear import create_infinicore_tensor
    device_str = "cpu"
    input_infini = create_infinicore_tensor(input, device_str)

    output = m.forward(input_infini)
    # print(m.weight_infini)
    # print(input_infini)
    print(output)


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


def func_net():
    modelpath = r"/home/ubuntu/Music/worksapce_nn/InfiniCore/test/infinicore/ops/model.pt"

    def test_TorchNet():
        import torch
        from torch import nn

        class TorchNet(nn.Module):
            def __init__(self):
                super(TorchNet, self).__init__()
                self.fc1 = nn.Linear(10, 6, bias=False)
                # self.relu = nn.ReLU()
                self.fc2 = nn.Linear(6, 1, bias=False)

            def forward(self, x):
                x = x.view((1, 10))
                output = self.fc2(self.fc1(x))
                return output

            def test(self):
                model = TorchNet()
                print(model)
                # torch.save(model.state_dict(), "model.pt")
                # print("-----> before \n", model.state_dict())

                model_param = torch.load(modelpath)
                model.load_state_dict(model_param)
                print("-----> after \n", model.state_dict())

                print('----------- caculate ------------>')
                x = torch.ones((1, 10), dtype=torch.float32)
                out = model.forward(x)
                print(out)

        TorchNet().test()

    def test_InfiniNet():
        import infinicore

        from infinicore import nn
        class InfiniNet(infinicore.nn.Module):
            def __init__(self):
                super(InfiniNet, self).__init__()
                self.fc1 = nn.Linear(10, 6, bias=False)
                self.fc2 = nn.Linear(6, 1, bias=False)

            def forward(self, x):
                x = x.view((1, 10))
                output = self.fc2(self.fc1(x))
                return output

            def test(self):
                model = InfiniNet()
                print(model)
                # torch.save(model.state_dict(), "model.pt")
                # print("-----> before \n", model.state_dict())
                import torch
                model_param = torch.load(modelpath)
                model.load_state_dict(model_param)
                # print("-----> after \n", model.state_dict())

                print('----------- caculate ------------>')

                device_str = "cuda"
                model.to(device=device_str)
                x = torch.ones((1, 10), dtype=torch.float32, device=device_str)

                out = model.forward(x)
                print(out)

                infini_x = infinicore.convert_torch_to_infini_tensor(x)

                out = model.forward(infini_x)
                print("==============>")
                print(out)

        InfiniNet().test()

    # test_TorchNet()
    test_InfiniNet()


def func7_mul():
    import infinicore

    import torch
    # x = torch.ones((2,3),device="cuda")*2
    # y = torch.ones((2,3),device="cuda")
    # x_infini = infinicore.convert_torch_to_infini_tensor(x)
    # y_infini = infinicore.convert_torch_to_infini_tensor(y)

    x_infini = infinicore.empty((2, 3), dtype=infinicore.float32, device=infinicore.device("cuda", 0))
    y_infini = infinicore.empty((2, 3), dtype=infinicore.float32, device=infinicore.device("cuda", 0))

    print("x_infini: ", x_infini)
    print("y_infini: ", y_infini)
    x_torch = infinicore.convert_infini_to_torch_tensor(x_infini)
    y_torch = infinicore.convert_infini_to_torch_tensor(y_infini)

    print("x_torch: ", x_torch)
    print("y_torch: ", y_torch)
    x_torch = torch.tensor([], dtype=x_torch.dtype, device=x_torch.device)
    xy_torch = torch.cat([x_torch, y_torch], -1)
    print("xy_torch: ", xy_torch)

    z_infini = x_infini + y_infini
    print("z_infini: ", z_infini)


def func8_test():
    import infinicore
    import torch

    data = torch.ones((2, 3))

    data_infini = infinicore.convert_torch_to_infini_tensor(data)
    print(data_infini)


def func():
    import infinicore

    from ctypes import POINTER, c_float, c_int, c_uint, c_void_p, byref, addressof

    data = [1, 2, 3, 4, 5]
    print()
    exit()

    print(type(data[1]))
    data_ptr = (c_int * len(data))(*data)

    address = addressof(data_ptr)

    ret = infinicore.from_blob(address,
                               [5],
                               dtype=infinicore.int32,
                               device=infinicore.device("cpu", 0))

    print(ret)


def func11():
    import numpy as np
    import ctypes
    import infinicore


    # ---
    data = list(range(0,5))

    ret = infinicore.convert_list_to_infini_tensor(data,shape=[2,3])

    print(ret)


    data = [1,2,2]
    print(torch.tensor(data))

    # ret_gpu = ret.to(infinicore.device("cuda", 0))
    # print(ret_gpu)

    # z_gpu = infinicore.empty((2, 3), dtype=infinicore.float32, device=infinicore.device("cuda", 0))
    # z_cpu = z_gpu.to(infinicore.device())
    # print(z_cpu)

    # ----------------
    # print("0000000000000000000000000000")
    # data = torch.ones((1,))
    # data = infinicore.convert_torch_to_infini_tensor(data)

    # value = infinicore.get_index_value(data,[-1])
    # print(value)


if __name__ == '__main__':
    # func7_mul()
    # func7_mul()
    func11()



 
