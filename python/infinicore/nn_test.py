



import infinicore


def func1():
    from infinicore import nn

    modelpath = r"/home/ubuntu/workspace_nn/InfiniCore_nn4/test/infinicore/ops/model.pt"

    class InfiniNet(infinicore.nn.Module):
        def __init__(self):
            super(InfiniNet, self).__init__()
            self.fc1 = nn.Linear(10, 6, bias=False)
            self.fc2 = nn.Linear(6, 1, bias=False)

        def forward(self, x):
            x = x.view((1, 10))
            output = self.fc2.forward(self.fc1.forward(x))
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

            device_str = "cpu"
            model.to(device=device_str)
            x = torch.ones((1, 10), dtype=torch.float32, device=device_str)

            out = model.forward(x)
            print(out)

            infini_x = infinicore.convert_torch_to_infini_tensor(x)

            out = model.forward(infini_x)
            print("==============>")
            print(out)

    InfiniNet().test()


if __name__ == '__main__':
    a = infinicore.nn.Linear(10,1)
