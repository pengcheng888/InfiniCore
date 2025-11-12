import infinicore
import torch


def func6():
    import infinicore

    input = torch.ones(5, 5)
    weight = torch.ones(5, 5)
    bias = torch.ones(
        5,
    )

    input_infini = infinicore.experimental.convert_torch_to_infini_tensor(input)
    weight_infini = infinicore.experimental.convert_torch_to_infini_tensor(weight)

    if bias is not None:
        bias_infini = infinicore.experimental.convert_torch_to_infini_tensor(bias)
    else:
        bias_infini = None

    y_infini = infinicore.nn.functional.linear(input_infini, weight_infini, bias_infini)

    print(y_infini)

    y_torch = torch.nn.functional.linear(input, weight, bias)

    print(y_torch)


def func7():
    import infinicore

    vocab_size = 10
    embedding_dim = 3

    input = torch.ones(1, 5, dtype=torch.int32)
    weight = torch.ones(vocab_size, embedding_dim)

    output = torch.ones(1, 5, embedding_dim)

    input_infini = infinicore.experimental.convert_torch_to_infini_tensor(input)
    weight_infini = infinicore.experimental.convert_torch_to_infini_tensor(weight)
    output_infini = infinicore.experimental.convert_torch_to_infini_tensor(output)

    y_infini = infinicore.nn.functional.embedding(
        input_infini, weight_infini, out=output_infini
    )

    print(y_infini)


def rotary_embedding(ans, t, sin, cos, device="cpu", algo=""):
    def _torch_rope(sin, cos, t1, t2):
        cos = cos.unsqueeze(1)  # [seq_len, 1, dh // 2]
        sin = sin.unsqueeze(1)  # [seq_len, 1, dh // 2]
        if device == "cpu":
            (t1, t2, cos, sin) = (
                t1.float(),
                t2.float(),
                cos.float(),
                sin.float(),
            )

        t_out_1 = t1 * cos - t2 * sin
        t_out_2 = t1 * sin + t2 * cos

        return t_out_1, t_out_2

    dh = t.shape[-1]
    dt = t.dtype
    assert dh % 2 == 0, "Embedding dimension must be even."

    if False:
        t_even = t[..., 0::2]  # [seq_len, n_head, dh // 2]
        t_odd = t[..., 1::2]  # [seq_len, n_head, dh // 2]

        t_out_even, t_out_odd = _torch_rope(sin, cos, t_even, t_odd)

        ans[..., 0::2] = t_out_even.to(dt)
        ans[..., 1::2] = t_out_odd.to(dt)
    else:
        half_dim = dh // 2
        t_first = t[..., :half_dim]
        t_second = t[..., half_dim:]

        t_out_first, t_out_second = _torch_rope(sin, cos, t_first, t_second)

        ans[..., :half_dim] = t_out_first.to(dt)
        ans[..., half_dim:] = t_out_second.to(dt)


def func8():
    from infinicore.lib import _infinicore

    print(_infinicore.Algo.GPT_NEOX)

    ntok, num, head_dim = 5, 32, 64
    ntok, num, head_dim = 1, 1, 64
    x = torch.ones((ntok, num, head_dim))
    pos_ids = torch.arange(0, x.shape[0], dtype=torch.int32)
    sin_table = torch.rand(ntok, head_dim // 2)
    cos_table = torch.rand(ntok, head_dim // 2)

    out = x.clone()
    rotary_embedding(
        out,
        x,
        sin_table,
        cos_table,
    )
    print(out)

    algo = _infinicore.Algo.GPT_NEOX

    x_infini = infinicore.experimental.convert_torch_to_infini_tensor(x)
    pos_ids_infini = infinicore.experimental.convert_torch_to_infini_tensor(pos_ids)
    sin_table_infini = infinicore.experimental.convert_torch_to_infini_tensor(sin_table)
    cos_table_infini = infinicore.experimental.convert_torch_to_infini_tensor(cos_table)

    y = infinicore.nn.functional.rope(
        x_infini, pos_ids_infini, sin_table_infini, cos_table_infini, algo
    )
    print(y)


if __name__ == "__main__":
    func8()
    pass
