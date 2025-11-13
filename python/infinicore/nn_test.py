import infinicore


def func3():
    import infinicore

    class InfiniNet(infinicore.nn.Module):
        def __init__(self):
            super(InfiniNet, self).__init__()
            self.rms = infinicore.nn.RMSNorm(normalized_shape=8)

        def forward(self, x: infinicore.Tensor):
            return self.rms(x)

    import torch

    class TorchNet(torch.nn.Module):
        def __init__(self):
            super(TorchNet, self).__init__()
            self.rms = torch.nn.RMSNorm(normalized_shape=8, device="cuda")

        def forward(self, x: torch.Tensor):
            return self.rms.forward(x)

    device_str = "cuda"

    weight = torch.ones((8,), device=device_str)

    model_param = {"rms.weight": infinicore.convert_torch_to_infini_tensor(weight)}

    model = InfiniNet()
    print(model)
    model.load_state_dict(model_param)

    # -------------- 构造输入数据 --------------
    input_torch = torch.ones((2, 8), device=device_str)
    input_infini = infinicore.convert_torch_to_infini_tensor(input_torch)

    y_infini = model.forward(input_infini)
    print(y_infini)
    # -------------------------------
    # -------------------------------
    model = TorchNet()
    print("weight", weight)
    model.load_state_dict({"rms.weight": weight})

    y_torch = model.forward(input_torch)

    print(y_torch)


def func4():
    import infinicore

    class InfiniNet(infinicore.nn.Module):
        def __init__(self):
            super(InfiniNet, self).__init__()
            self.emb = infinicore.nn.Embedding(5, 5)

        def forward(self, x: infinicore.Tensor):
            return self.emb.forward(x)

    import torch

    class TorchNet(torch.nn.Module):
        def __init__(self):
            super(TorchNet, self).__init__()
            self.emb = torch.nn.Embedding(5, 5)

        def forward(self, x: torch.Tensor):
            return self.emb.forward(x)

    device_str = "cpu"

    weight = torch.ones((5, 5), device=device_str)

    model_param = {"emb.weight": infinicore.convert_torch_to_infini_tensor(weight)}

    model = InfiniNet()
    print(model)

    model.load_state_dict(model_param)

    # -------------- 构造输入数据 --------------
    input_torch = torch.ones((1, 2), dtype=torch.int64, device=device_str)
    input_infini = infinicore.convert_torch_to_infini_tensor(input_torch)

    y_infini = model.forward(input_infini)
    print(y_infini)

    # -------------------------------
    # -------------------------------
    model = TorchNet()

    model.load_state_dict({"emb.weight": weight})

    y_torch = model.forward(input_torch)

    print(y_torch)


def func5():
    import infinicore

    class InfiniNet(infinicore.nn.Module):
        def __init__(self):
            super(InfiniNet, self).__init__()

            self.layers = infinicore.nn.ModuleList(
                [infinicore.nn.Linear(5, 5), infinicore.nn.Linear(5, 5)]
            )

        def forward(self, x: infinicore.Tensor):
            for layer in self.layers:
                x = layer.forward(x)
            return x

    import torch

    class TorchNet(torch.nn.Module):
        def __init__(self):
            super(TorchNet, self).__init__()

            self.layers = torch.nn.ModuleList(
                [torch.nn.Linear(5, 5, bias=False), torch.nn.Linear(5, 5, bias=False)]
            )

        def forward(self, x: torch.Tensor):
            for layer in self.layers:
                x = layer(x)
            return x

    params = {"layers.0.weight": torch.ones(5, 5), "layers.1.weight": torch.ones(5, 5)}

    if False:
        model = TorchNet()
        for k, v in model.named_parameters():
            print(k)

        model.load_state_dict(params)

        input = torch.ones(5, 5)
        y = model(input)
        print(y)

    if True:

        def torch_2_infini_ref(model_param: dict):
            print("model_param: ", id(model_param))

            model_param_infini = {}
            for key, value in model_param.items():
                model_param_infini[key] = (
                    infinicore.experimental.torch_2_infini_tensor_ref(value)
                )

            return model_param_infini

        model = InfiniNet()

        params_infini = torch_2_infini_ref(params)

        model.load_state_dict(params_infini)

        input = torch.ones(5, 5)
        infini_x = infinicore.convert_torch_to_infini_tensor(input)

        infini_y = model.forward(infini_x)
        print(infini_y)


def func6():
    import infinicore
    import torch

    input = torch.ones(5, 5)
    # weight=   infinicore.empty((2,2), dtype=infinicore.float32, device=infinicore.device("cpu", 0))
    # bias=   infinicore.empty((2,), dtype=infinicore.float32, device=infinicore.device("cpu", 0))
    weight = torch.ones(5, 5)
    bias = torch.ones(
        5,
    )

    input_infini = infinicore.convert_torch_to_infini_tensor(input)
    weight_infini = infinicore.convert_torch_to_infini_tensor(weight)
    bias_infini = infinicore.convert_torch_to_infini_tensor(bias)

    y_infini = infinicore.nn.functional.linear(input_infini, weight_infini, bias_infini)

    print(y_infini)

    y_torch = torch.nn.functional.linear(input, weight, bias)

    print(y_torch)


def func7():
    import torch
    import torch.nn.functional as F
    import math

    # Efficient implementation equivalent to the following:
    def scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(
                diagonal=0
            )
            print(temp_mask)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            print(attn_bias)
            exit(-1)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value

    print()
    print()

    device_str = "cuda"
    # Optionally use the context manager to ensure one of the fused kernels is run
    # query = torch.rand(1, 8, 128, 64, dtype=torch.float16, device=device_str)
    # key = torch.rand(1, 8, 128, 64, dtype=torch.float16, device=device_str)
    # value = torch.rand(1, 8, 128, 64, dtype=torch.float16, device=device_str)

    # query = torch.rand(1, 32, 10, 64, dtype=torch.float32, device=device_str)
    # key = torch.rand(1, 8, 128, 64, dtype=torch.float32, device=device_str)
    # value = torch.rand(1, 8, 128, 64, dtype=torch.float32, device=device_str)

    query = torch.rand(1, 32, 2, 64, dtype=torch.float32, device=device_str)
    key = torch.rand(1, 8, 11, 64, dtype=torch.float32, device=device_str)
    value = torch.rand(1, 8, 11, 64, dtype=torch.float32, device=device_str)

    out1 = F.scaled_dot_product_attention(
        query, key, value, is_causal=True, enable_gqa=True
    )
    print(out1.shape)

    out2 = scaled_dot_product_attention(
        query, key, value, is_causal=True, enable_gqa=True
    )
    print(out2.shape)

    print("torch.abs(out1 - out2).max() ", torch.abs(out1 - out2).max())
    # print(torch.abs(out1 - out2))
    # -------------------------------------------------------------------- #
    query_infini = infinicore.from_torch(query)
    key_infini = infinicore.from_torch(key)
    value_infini = infinicore.from_torch(value)
    out_infini = infinicore.nn.functional.scaled_dot_product_attention(
        query_infini, key_infini, value_infini, is_causal=True, enable_gqa=True
    )
    print(out_infini.shape)

    out_torch = infinicore.convert_infini_to_torch_tensor(out_infini)
    print("torch.abs(out_torch - out2).max() ", torch.abs(out_torch - out2).max())
    print(
        "torch.abs(out_torch - out1).max() ",
        torch.abs(out_torch - out1).max(),
    )


def func8():
    from infinicore import nn

    class InfiniNet(infinicore.nn.Module):
        def __init__(self):
            super(InfiniNet, self).__init__()

            self.weight = infinicore.nn.Parameter(
                infinicore.empty(
                    (1,), dtype=infinicore.float32, device=infinicore.device("cpu", 0)
                )
            )
            print("?>>", self.weight)

        def forward(self):
            pass

        def test(self):
            model = InfiniNet()
            print(model)

    InfiniNet().test()


if __name__ == "__main__":
    func3()
