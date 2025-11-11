import infinicore
def func1():
    from infinicore import nn

    modelpath = r"/home/ubuntu/Music/nn5_my/InfiniCore/test/infinicore/ops/model.pt"

    class InfiniNet(infinicore.nn.Module):
        def __init__(self):
            super(InfiniNet, self).__init__()
            self.num = 10
            self.fc1 = infinicore.nn.Linear(10, 6, bias=False)
            self.fc2 = infinicore.nn.Linear(6, 1, bias=False)

        def forward(self, x):
            x = x.view((1, 10))
            output = self.fc2.forward(self.fc1.forward(x))
            print()
            return output

        def test(self):
            model = InfiniNet()
            print(model)
            # torch.save(model.state_dict(), "model.pt")
            # print("-----> before \n", model.state_dict())
            import torch
            model_param = torch.load(modelpath,map_location=torch.device("cuda"))
          
            def torch_2_infini_ref(model_param:dict):
                print("model_param: ", id(model_param) )

                model_param_infini = {}
                for key,value in model_param.items():
                    model_param_infini[key] = infinicore.experimental.torch_2_infini_tensor_ref(value)
                
                return model_param_infini
  
            
            print("model_param: ", id(model_param) )
            model_param_infini = torch_2_infini_ref(model_param)
            
            print("model_param: ", id(model_param) )
 
            model.load_state_dict(model_param_infini)
            # print("-----> after \n", model.state_dict())
   
    
            # print(  model.fc1)
            # print(  model.num)
            
           
            print('----------- caculate ------------>')

            device_str = "cuda"
       
            x = torch.ones((1, 10), dtype=torch.float32, device=device_str)

            # out = model.forward(x)
            # print(out)
        
            infini_x = infinicore.convert_torch_to_infini_tensor(x)
    
            out = model.forward(infini_x)

            print(out)

    # infinicore.nn.Linear(10, 6, bias=False)
    # exit(-1)

    InfiniNet().test()

def func3():
    import infinicore
    class InfiniNet(infinicore.nn.Module):
        def __init__(self):
            super(InfiniNet, self).__init__()
            self.rms = infinicore.nn.RMSNorm(normalized_shape=8)

        def forward(self, x:infinicore.Tensor):
            return self.rms.forward(x)
    
    import torch
    class TorchNet(torch.nn.Module):
        def __init__(self):
            super(TorchNet, self).__init__()
            self.rms = torch.nn.RMSNorm(normalized_shape=8,device="cuda")

        def forward(self, x:torch.Tensor):
            return self.rms.forward(x)
        

    device_str = "cuda"
    
    weight = torch.ones((8,),device=device_str)

    model_param = {
        "rms.weight": infinicore.convert_torch_to_infini_tensor(weight)
    }

    model = InfiniNet()
    print(model)

    model.load_state_dict(model_param)

    # -------------- 构造输入数据 --------------
    input_torch = torch.ones((2,8),device=device_str)
    input_infini = infinicore.convert_torch_to_infini_tensor(input_torch)

    y_infini = model.forward(input_infini)
    print(y_infini)
    # -------------------------------
    # -------------------------------
    model = TorchNet()
    print("weight",weight)
    model.load_state_dict(  {  "rms.weight": weight })
    
    y_torch = model.forward(input_torch)

    print(y_torch)


def func4():
    import infinicore
    class InfiniNet(infinicore.nn.Module):
        def __init__(self):
            super(InfiniNet, self).__init__()
            self.emb = infinicore.nn.Embedding(5,5)

        def forward(self, x:infinicore.Tensor):
            return self.emb.forward(x)
    
    import torch
    class TorchNet(torch.nn.Module):
        def __init__(self):
            super(TorchNet, self).__init__()
            self.emb = torch.nn.Embedding(5,5)

        def forward(self, x:torch.Tensor):
            return self.emb.forward(x)
        

    device_str = "cpu"
    
    weight = torch.ones((5,5),device=device_str)

    model_param = {
        "emb.weight": infinicore.convert_torch_to_infini_tensor(weight)
    }

    model = InfiniNet()
    print(model)

    model.load_state_dict(model_param)

    # -------------- 构造输入数据 --------------
    input_torch = torch.ones((1,2),dtype=torch.int64,device=device_str)
    input_infini = infinicore.convert_torch_to_infini_tensor(input_torch)

    y_infini = model.forward(input_infini)
    print(y_infini)

    # -------------------------------
    # -------------------------------
    model = TorchNet()

    model.load_state_dict(  {  "emb.weight": weight })
    
    y_torch = model.forward(input_torch)

    print(y_torch)


def func5():
    import infinicore
    class InfiniNet(infinicore.nn.Module):
        def __init__(self):
            super(InfiniNet, self).__init__()

            self.layers = infinicore.nn.ModuleList(
            [infinicore.nn.Linear(5,5),
             infinicore.nn.Linear(5,5)]
            )

        def forward(self, x:infinicore.Tensor):

            for layer in self.layers:
                x = layer.forward(x)
            return x 
    
    import torch
    class TorchNet(torch.nn.Module):
        def __init__(self):
            super(TorchNet, self).__init__()

            self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(5,5, bias=False),
             torch.nn.Linear(5,5, bias=False)]
            )

        def forward(self, x:torch.Tensor):

            for layer in self.layers:
                x = layer(x)
            return x 
    
    params = {"layers.0.weight":torch.ones(5,5),
             "layers.1.weight":torch.ones(5,5)}
    
    if False:
        model = TorchNet()
        for k,v in model.named_parameters():
            print(k)
        
        model.load_state_dict( params)

        input = torch.ones(5,5)
        y = model(input)
        print(y)

    if True:
        def torch_2_infini_ref(model_param:dict):
            print("model_param: ", id(model_param) )

            model_param_infini = {}
            for key,value in model_param.items():
                model_param_infini[key] = infinicore.experimental.torch_2_infini_tensor_ref(value)
            
            return model_param_infini
        
        model = InfiniNet()


        params_infini = torch_2_infini_ref(params)
        
        model.load_state_dict( params_infini)

        input = torch.ones(5,5)
        infini_x = infinicore.convert_torch_to_infini_tensor(input)
        
        infini_y = model.forward(infini_x)
        print(infini_y)

def func6():
    import infinicore

    import torch
    

    input = torch.ones(5,5) 
    # weight=   infinicore.empty((2,2), dtype=infinicore.float32, device=infinicore.device("cpu", 0))
    # bias=   infinicore.empty((2,), dtype=infinicore.float32, device=infinicore.device("cpu", 0))
    weight = torch.ones(5,5) 
    bias = torch.ones(5,) 


    input_infini = infinicore.convert_torch_to_infini_tensor(input)
    weight_infini = infinicore.convert_torch_to_infini_tensor(weight)
    bias_infini = infinicore.convert_torch_to_infini_tensor(bias)

    
    y_infini  = infinicore.nn.functional.linear(input_infini, weight_infini, bias_infini)

    print(y_infini)

    y_torch = torch.nn.functional.linear(input, weight, bias)

    print(y_torch)


if __name__ == '__main__':
    func5()
