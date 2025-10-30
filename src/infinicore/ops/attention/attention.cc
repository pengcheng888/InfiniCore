#include "infinicore/ops/attention.hpp"
#include "infinicore/ops/linear.hpp"

namespace infinicore::op {

common::OpDispatcher<Attention::schema> &Attention::dispatcher() {
    static common::OpDispatcher<Attention::schema> dispatcher_;
    return dispatcher_;
};

void Attention::execute(Tensor out, Tensor q, Tensor k, Tensor v, Tensor k_cache, Tensor v_cache, size_t pos) {
    dispatcher().lookup(context::getDevice().getType())(out, q, k, v, k_cache, v_cache, pos);
}

Tensor attention(Tensor q, Tensor k, Tensor v, Tensor k_cache, Tensor v_cache, size_t pos) {
    size_t n_q_head = q->shape()[0];
    size_t seq_len = q->shape()[1];
    size_t head_dim = q->shape()[2];
    Shape shape = {seq_len, n_q_head, head_dim};
    auto out = Tensor::empty(shape, q->dtype(), q->device());
    attention_(out, q, k, v, k_cache, v_cache, pos);
    return out;
}

void attention_(Tensor out, Tensor q, Tensor k, Tensor v, Tensor k_cache, Tensor v_cache, size_t pos) {
    Attention::execute(out, q, k, v, k_cache, v_cache, pos);
}

void causalSoftmax(Tensor out, Tensor in) {
    ;
}

Tensor attention_lm(Tensor query_states, // [bs, num_attention_heads, ntoken, head_dim]
                    Tensor key_states,   // [bs, num_key_value_heads, total_token, head_dim]
                    Tensor value_states, // [bs, num_key_value_heads, total_token, head_dim]
                    float attention_scale) {

    auto query_shape = query_states->shape();
    auto key_shape = key_states->shape();
    auto value_shape = value_states->shape();

    Size batch_size = query_shape[0];
    Size num_attention_heads = query_shape[1];
    Size ntoken = query_shape[2];

    Size num_key_value_heads = key_shape[1];
    Size total_token = key_shape[2];
    Size head_dim = key_shape[3];

    assert(0 == (num_attention_heads % num_key_value_heads));
    Size ngroup = num_attention_heads / num_key_value_heads;

    Tensor attn_values = Tensor::empty({batch_size, num_key_value_heads, ngroup, ntoken, head_dim}, query_states->dtype(), query_states->device());
    for (Size ib = 0; ib < batch_size; ++ib) {
        Tensor q = query_states->narrow({{0, ib, 1}}); // [1, num_attention_heads, ntoken, head_dim]
        Tensor k = key_states->narrow({{0, ib, 1}});   // [1, num_key_value_heads, total_token, head_dim]
        Tensor v = value_states->narrow({{0, ib, 1}}); // [1, num_key_value_heads, total_token, head_dim]

        // 变换
        Tensor q_gemm = k->view({num_attention_heads, ntoken, head_dim})->permute({1, 0, 2})->view({ntoken, num_key_value_heads, ngroup, head_dim})->permute({1, 2, 0, 3}); // => { num_key_value_heads, ngroup, ntoken, head_dim}
        Tensor k_gemm = k->view({num_key_value_heads, total_token, head_dim})->permute({0, 2, 1});
        Tensor v_gemm = k->view({num_key_value_heads, total_token, head_dim});

        // 计算
        // rearrange_q_buf 是 {nkvh, ngroup * max_seq_len, dh}， 跟 q_rearrange 共享地址
        // 矩阵乘法：    {nkvh, ngroup * seq_len, dh} @ { nkvh, dh, total_len}  ==>  {nkvh, ngroup * seq_len, total_len}
        Tensor qk_scores = Tensor::empty({num_key_value_heads, ngroup * ntoken, total_token}, query_states->dtype(), query_states->device());
        // linear(qk_scores,
        //        q_gemm->view({num_key_value_heads, ngroup * ntoken, head_dim}), // {num_key_value_heads, ngroup * ntoken, head_dim}
        //        k_gemm,                                                         // {num_key_value_heads, head_dim, total_token}
        //        attention_scale,
        //        0.f, nullptr, nullptr);

        Tensor qk_softmax = qk_scores->view({num_attention_heads, ntoken, total_token});
        causalSoftmax(qk_softmax, qk_softmax);

        Tensor attn_val = attn_values->narrow({{0, ib, 1}}); //{1, num_key_value_heads, ngroup, ntoken, head_dim}
        // linear(attn_val->view({num_key_value_heads, ngroup * ntoken, head_dim}),
        //        qk_scores,
        //        v_gemm, 1.f, 0.f, nullptr, nullptr);
    }

    return query_states;
}

} // namespace infinicore::op
