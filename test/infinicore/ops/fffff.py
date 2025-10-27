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


def func_net():

    modelpath =  r"/home/ubuntu/workspace_nn/InfiniCore/test/infinicore/ops/model.pt"
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
                print("-----> after \n", model.state_dict())

                print('----------- caculate ------------>')
                from infinicore.nn.modules.linear import create_infinicore_tensor, print_infini_tensor
                device_str = "cpu"
                x = torch.ones((1, 10), dtype=torch.float32)

                out = model.forward(x)
                print(out)

                infini_x = create_infinicore_tensor(x, device_str)

                out = model.forward(infini_x)
                print_infini_tensor(out)

        InfiniNet().test()

    #test_TorchNet()
    test_InfiniNet()


def test6():
    import infinicore
    from infinicore import nn
    infinicore.nn.RMSNorm.testop()


def func7_mul():
    import infinicore
    from infinicore.nn.modules.linear import torch_tensor_2_infini_tensor
    import torch
    x = torch.ones((2,3))*2
    y = torch.ones((2,3))
    x_infini = torch_tensor_2_infini_tensor(x, device_str="cpu")
    y_infini = torch_tensor_2_infini_tensor(y, device_str="cpu")

    z_infini = x_infini * y_infini
    print(y_infini)
    print(y_infini)
    print(z_infini)

def func8_test():
    import infinicore
    from infinicore.nn.modules.linear import torch_tensor_2_infini_tensor,print_infini_tensor
    import torch

    data = torch.ones((2,3))

    data_infini = torch_tensor_2_infini_tensor(data, device_str="cpu")
    print(data_infini)

    print_infini_tensor(data_infini)
    

if __name__ == '__main__':
    func7_mul()

