import torch
from torch import nn
import numpy as np
import infinicore

torch.manual_seed(10)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dim=None, qk_dim=None, head_num=2):
        super().__init__()
        #  Enzo_Mi  https://www.bilibili.com/video/BV1gV4y167rS
        self.dim = dim
        self.head_num = head_num
        self.dim_head = qk_dim // head_num
        assert qk_dim % head_num == 0

        self.WQ = nn.Linear(dim, qk_dim)
        self.WK = nn.Linear(dim, qk_dim)
        self.WV = nn.Linear(dim, qk_dim)
        self.lin = nn.Linear(qk_dim, qk_dim)
        self.scale = 1 / np.sqrt(self.dim_head)

    def forward(self, X):
        bs, n, dim = X.shape

        Q = self.WQ(X)
        K = self.WK(X)
        V = self.WV(X)

        Q = Q.reshape((bs, n, self.head_num, self.dim_head)).transpose(1, 2)
        K = K.reshape((bs, n, self.head_num, self.dim_head)).transpose(1, 2)
        V = V.reshape((bs, n, self.head_num, self.dim_head)).transpose(1, 2)

        score = Q @ K.transpose(2, 3) * self.scale
        score = torch.softmax(score, dim=-1)

        out = score @ V
        out = out.transpose(1, 2).reshape(bs, n, dim)

        return self.lin(out)

    def test():
        data = torch.rand((1, 3, 8), dtype=torch.float32)  # bs, n, dim
        print(data)

        bs, n, dim = data.shape

        att = MultiHeadAttention(dim, 4, dim)

        out = att(data)
        print(out)


def cac(
    query_states,  # [bs, num_attention_heads,  seq_len,   head_dim]
    key_states,  # [bs, num_key_value_heads, total_seq_len, head_dim]
    value_states,  # [bs, num_key_value_heads, total_seq_len, head_dim]
):
    # query,  # (1,32, 5,64)
    # key,  # (1,4*8,9,64)
    # value,  # (1,4*8,9,64)

    bs, num_attention_heads, seq_len, head_dim = query_states.shape
    _, num_key_value_heads, total_seq_len, _ = key_states.shape

    Q = query_states
    K = key_states
    V = value_states

    scale = 1 / np.sqrt(head_dim)

    print(scale)

    ##

    # 创建因果掩码（下三角矩阵）
    mask = torch.tril(
        torch.ones(1, 1, seq_len, total_seq_len, device=query_states.device)
    )

    score = (
        Q @ K.transpose(2, 3) * scale
    )  # ==> [bs, num_attention_heads, seq_len,  total_seq_len]

    score = score.masked_fill(mask == 0, -1e9)

    score = torch.softmax(
        score, dim=-1
    )  # ==> [bs, num_attention_heads,  seq_len,  total_seq_len]

    out = score @ V  # ==> [bs, num_attention_heads,  seq_len,   head_dim]

    return out


def func(
    query_states,  # [bs,  seq_len,       num_attention_heads, head_dim]
    key_states,  #   [ bs, total_seq_len, num_key_value_heads, head_dim]
    value_states,  # [ bs, total_seq_len, num_key_value_heads, head_dim]
):
    bs, num_attention_heads, seq_len, head_dim = 1, 32, 5, 64
    total_seq_len = 9
    num_key_value_heads = num_attention_heads

    query_states = torch.rand((bs, num_attention_heads, seq_len, head_dim))
    key_states = torch.rand((bs, num_key_value_heads, total_seq_len, head_dim))
    value_states = torch.rand((bs, num_key_value_heads, total_seq_len, head_dim))

    # query_states = torch.load("query.pt").to(dtype=torch.float32)
    # key_states = torch.load("key.pt").to(dtype=torch.float32)
    # value_states = torch.load("value.pt").to(dtype=torch.float32)
    # attn_output = torch.load("attn_output.pt").to(dtype=torch.float32)

    out = cac(
        query_states,
        key_states,
        value_states,
    )

    print("----------------------------------------")
    print(out.shape, out)


# Efficient implementation equivalent to the following:
# def scaled_dot_product_attention(
#     query, key, value, is_causal=True, enable_gqa=True
# ) -> torch.Tensor:
#     L, S = query.size(-3), key.size(-3)
#     scale_factor = 1 / np.sqrt(query.size(-1))
#     attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

#     if is_causal:
#         temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
#         attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

#     if enable_gqa:
#         key = key.repeat_interleave(query.size(-2) // key.size(-2), -2)
#         value = value.repeat_interleave(query.size(-2) // value.size(-2), -2)

#     # => [ num_attention_heads, seq_len,       head_dim]
#     query = query.permute((1, 0, 2))
#     # => [ num_key_value_heads, total_seq_len, head_dim]
#     key = key.permute((1, 0, 2))
#     # => [ num_key_value_heads, total_seq_len, head_dim]
#     value = value.permute((1, 0, 2))

#     attn_weight = query @ key.transpose(-2, -1) * scale_factor
#     attn_weight += attn_bias
#     attn_weight = torch.softmax(attn_weight, dim=-1)

#     return attn_weight @ value


# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / np.sqrt(query.size(-1))
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if True:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if True:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    return attn_weight @ value


def func2():
    def torch_attention(
        query_states,  # (1,5,32, 64)
        key_states,  #   (1,9,4*8,64)
        value_states,  # (1,9,4*8,64)
    ):
        seq_len, num_attention_heads, head_dim = query_states.shape
        total_seq_len, num_key_value_heads, _ = key_states.shape

        # => [ num_attention_heads, seq_len,       head_dim]
        Q = query_states.transpose(0, 1)
        # => [ num_key_value_heads, total_seq_len, head_dim]
        K = key_states.transpose(0, 1)
        # => [ num_key_value_heads, total_seq_len, head_dim]
        V = value_states.transpose(0, 1)

        scale = 1 / np.sqrt(head_dim)

        # [ num_attention_heads, seq_len, head_dim] @ [ num_key_value_heads, head_dim, total_seq_len]
        # => [ num_attention_heads, seq_len, total_seq_len]
        score = Q @ K.transpose(1, 2) * scale

        mask = torch.tril(
            torch.ones(1, seq_len, total_seq_len, device=query_states.device)
        )
        score = score.masked_fill(mask == 0, -1e9)

        score = torch.softmax(score, dim=-1)

        # exit(-1)

        # [ num_attention_heads,  seq_len,  total_seq_len] @  [ num_key_value_heads, total_seq_len, head_dim]
        # => [ num_attention_heads,  seq_len,   head_dim]
        out = score @ V

        # => [  seq_len, num_attention_heads, head_dim]
        out = out.transpose(0, 1).contiguous()
        return out

    def infini_attention(
        query_states: infinicore.Tensor,
        key_states: infinicore.Tensor,
        value_states: infinicore.Tensor,
    ):
        seq_len, num_attention_heads, head_dim = query_states.shape
        total_seq_len, num_key_value_heads, _ = key_states.shape

        # => [ num_attention_heads, seq_len,       head_dim]
        Q = query_states.permute((1, 0, 2))
        # => [ num_key_value_heads, total_seq_len, head_dim]
        K = key_states.permute((1, 0, 2))
        # => [ num_key_value_heads, total_seq_len, head_dim]
        V = value_states.permute((1, 0, 2))

        scale = 1 / np.sqrt(head_dim)

        # [ num_attention_heads, seq_len, head_dim] @ [ num_key_value_heads, head_dim, total_seq_len]
        # => [ num_attention_heads, seq_len, total_seq_len]

        scale = infinicore.from_list([scale], dtype=Q.dtype).as_strided(
            ([num_attention_heads, seq_len, total_seq_len]), [0, 0, 0]
        )

        score = Q @ K.permute((0, 2, 1)) * scale

        infinicore.nn.functional.causal_softmax(score, out=score)

        # [ num_attention_heads,  seq_len,  total_seq_len] @   num_key_value_heads, total_seq_len, head_dim]
        # => [ num_attention_heads,  seq_len,   head_dim]
        out = score @ V

        # => [seq_len, num_attention_heads, head_dim]
        out = out.permute((1, 0, 2)).contiguous()
        return out

    seq_len, num_attention_heads, head_dim = 2, 3, 4
    total_seq_len, num_key_value_heads, _ = 2, 3, 4

    query_states = torch.rand(
        (seq_len, num_attention_heads, head_dim), dtype=torch.float32
    )
    key_states = torch.rand(
        (total_seq_len, num_key_value_heads, head_dim), dtype=torch.float32
    )
    value_states = torch.rand(
        (total_seq_len, num_key_value_heads, head_dim), dtype=torch.float32
    )

    print("------------------------------------------------")
    out = torch_attention(query_states, key_states, value_states)
    print(out.shape, out)

    print("------------------------------------------------")
    query_states_infini = query_states.to_infini()
    key_states_infini = key_states.to_infini()
    value_states_infini = value_states.to_infini()
    out_infini = infini_attention(
        query_states_infini, key_states_infini, value_states_infini
    )
    print(out_infini.shape, out_infini)

    print("------------------------------------------")

    out_s = scaled_dot_product_attention(query_states, key_states, value_states)
    print(out_s.shape, out_s)


def func3():
    def infini_attention(
        query_states: infinicore.Tensor,
        key_states: infinicore.Tensor,
        value_states: infinicore.Tensor,
    ):
        seq_len, num_attention_heads, head_dim = query_states.shape
        total_seq_len, num_key_value_heads, _ = key_states.shape
        print(key_states.shape, key_states)
        if False:
            key_states_new = infinicore.empty_like(query_states)
            value_states_new = infinicore.empty_like(query_states)
            ngroup = num_attention_heads // num_key_value_heads
            for i in range(ngroup):
                key_states_new.narrow(
                    1, i * num_key_value_heads, num_key_value_heads
                ).copy_(key_states)

                value_states_new.narrow(
                    1, i * num_key_value_heads, num_key_value_heads
                ).copy_(value_states)
            key_states = key_states_new
            value_states = value_states_new
        else:
            print(key_states)

            ngroup = num_attention_heads // num_key_value_heads
            key_states_new = infinicore.empty_like(query_states).view(
                (total_seq_len, num_key_value_heads, ngroup, head_dim)
            )
            value_states_new = infinicore.empty_like(query_states).view(
                (total_seq_len, num_key_value_heads, ngroup, head_dim)
            )
            for i in range(ngroup):
                key_states_new.narrow(2, i, 1).copy_(
                    key_states.view((total_seq_len, num_key_value_heads, 1, head_dim))
                )

                value_states_new.narrow(2, i, 1).copy_(
                    value_states.view((total_seq_len, num_key_value_heads, 1, head_dim))
                )
            key_states = key_states_new.contiguous().view(
                (total_seq_len, num_key_value_heads * ngroup, head_dim)
            )

            value_states = value_states_new.contiguous().view(
                (total_seq_len, num_key_value_heads * ngroup, head_dim)
            )

        # => [ num_attention_heads, seq_len,       head_dim]
        Q = query_states.permute((1, 0, 2))
        # => [ num_key_value_heads, total_seq_len, head_dim]
        K = key_states.permute((1, 0, 2))
        # => [ num_key_value_heads, total_seq_len, head_dim]
        V = value_states.permute((1, 0, 2))

        scale = 1 / np.sqrt(head_dim)

        # [ num_attention_heads, seq_len, head_dim] @ [ num_key_value_heads, head_dim, total_seq_len]
        # => [ num_attention_heads, seq_len, total_seq_len]

        scale = infinicore.from_list([scale], dtype=Q.dtype).as_strided(
            ([num_attention_heads, seq_len, total_seq_len]), [0, 0, 0]
        )

        score = Q @ K.permute((0, 2, 1)) * scale

        infinicore.nn.functional.causal_softmax(score, out=score)

        # [ num_attention_heads,  seq_len,  total_seq_len] @   num_key_value_heads, total_seq_len, head_dim]
        # => [ num_attention_heads,  seq_len,   head_dim]
        out = score @ V

        # => [seq_len, num_attention_heads, head_dim]
        out = out.permute((1, 0, 2)).contiguous()
        return out

    # seq_len, num_attention_heads, head_dim = 1, 32, 64
    # total_seq_len, num_key_value_heads, _ = 6, 4, 64
    seq_len, num_attention_heads, head_dim = 1, 6, 4
    total_seq_len, num_key_value_heads, _ = 2, 3, 4
    query_states = torch.rand(
        (seq_len, num_attention_heads, head_dim), dtype=torch.float32
    )
    key_states = torch.rand(
        (total_seq_len, num_key_value_heads, head_dim), dtype=torch.float32
    )
    value_states = torch.rand(
        (total_seq_len, num_key_value_heads, head_dim), dtype=torch.float32
    )

    query_states_infini = query_states.to_infini()
    key_states_infini = key_states.to_infini()
    value_states_infini = value_states.to_infini()
    out_infini = infini_attention(
        query_states_infini, key_states_infini, value_states_infini
    )
    print(out_infini.shape, out_infini)

    print("------------------------------------------")
    query_states = (
        query_states.view((1, seq_len, num_attention_heads, head_dim))
        .permute((0, 2, 1, 3))
        .contiguous()
    )
    key_states = (
        key_states.view((1, total_seq_len, num_key_value_heads, head_dim))
        .permute((0, 2, 1, 3))
        .contiguous()
    )
    value_states = (
        value_states.view((1, total_seq_len, num_key_value_heads, head_dim))
        .permute((0, 2, 1, 3))
        .contiguous()
    )

    out_s = scaled_dot_product_attention(query_states, key_states, value_states)
    print(out_s.shape, out_s)


if __name__ == "__main__":
    # func3()

    A = infinicore.empty(
        [1, 32, 5, 5],
        dtype=infinicore.float16,
        device=infinicore.device("cuda", 0),
    )
    B = infinicore.empty(
        [1, 32, 5, 64],
        dtype=infinicore.float16,
        device=infinicore.device("cuda", 0),
    )

    print(A @ B)
