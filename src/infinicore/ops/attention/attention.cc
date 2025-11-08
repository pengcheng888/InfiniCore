#include "infinicore/ops/attention.hpp"
#include "infinicore/ops/causal_softmax.hpp"
#include "infinicore/ops/gemm.hpp"

#include <cmath>
// #include <math.h>
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

Tensor attention_InfiniLM(Tensor q_states, // { nh, seq_len, dh}
                          Tensor k_states, // { nkvh,total_len, dh}
                          Tensor v_states, // { nkvh,total_len, dh}
                          float attention_scale,
                          size_t dh,
                          size_t nh,
                          size_t nkvh,
                          size_t ngroup,
                          size_t seq_len,
                          size_t total_len) {

    auto q_gemm = q_states->view({nkvh, ngroup * seq_len, dh}); // {nkvh, ngroup * seq_len, dh}
    auto k_gemm = k_states->permute({0, 2, 1});                 // { nkvh, dh, total_len}
    auto v_gemm = v_states;                                     // { nkvh, total_len, dh}

    // printf(" ----------- 11 \n");
    // printf(" ----------- dh  %ld \n", dh);
    // printf(" ----------- nh  %ld \n", nh);
    // printf(" ----------- nkvh  %ld \n", nkvh);

    // printf(" ----------- ngroup  %ld \n", ngroup);
    // printf(" ----------- seq_len  %ld \n", seq_len);
    // printf(" ----------- total_len  %ld \n", total_len);

    // --------------------------------------
    // qk_score : {nkvh, ngroup * seq_len, total_len}
    Tensor qk_score = gemm(q_gemm, // {nkvh, ngroup * seq_len, dh}
                           k_gemm, // {nkvh, dh, total_len}
                           attention_scale, 0.f);

    // softmax
    auto qk_softmax = qk_score->view({nh, seq_len, total_len});
    causal_softmax_(qk_softmax, qk_softmax);

    // values
    // att_val : {nkvh, ngroup * seq_len, dh}
    Tensor att_val = gemm(qk_score, // {nkvh, ngroup * seq_len, total_len}
                          v_gemm,   // { nkvh, total_len, dh}
                          1.0f,
                          0.0f);

    return att_val;
}

Tensor attention_lm(Tensor query_states, // [bs, num_attention_heads, ntoken, head_dim]
                    Tensor key_states,   // [bs, num_key_value_heads, total_token, head_dim]
                    Tensor value_states  // [bs, num_key_value_heads, total_token, head_dim]
) {

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

    float attention_scale = 1.f / float(sqrt(head_dim));

    // printf(" ----------- 11 \n");
    // printf(" ----------- batch_size  %ld \n", batch_size);
    // printf(" ----------- num_attention_heads  %ld \n", num_attention_heads);
    // printf(" ----------- ntoken  %ld \n", ntoken);

    // printf(" ----------- num_key_value_heads  %ld \n", num_key_value_heads);
    // printf(" ----------- total_token  %ld \n", total_token);
    // printf(" ----------- head_dim  %ld \n", head_dim);

    // printf(" ----------- ngroup  %ld \n", ngroup);

    for (Size ib = 0; ib < batch_size; ++ib) {
        Tensor q = query_states->narrow({{0, ib, 1}})->view({num_attention_heads, ntoken, head_dim});      // [ num_attention_heads, ntoken, head_dim]
        Tensor k = key_states->narrow({{0, ib, 1}})->view({num_key_value_heads, total_token, head_dim});   // [ num_key_value_heads, total_token, head_dim]
        Tensor v = value_states->narrow({{0, ib, 1}})->view({num_key_value_heads, total_token, head_dim}); // [ num_key_value_heads, total_token, head_dim]

        // att_val : {nkvh, ngroup * seq_len, dh}
        Tensor att_val = attention_InfiniLM(q, // { nh, seq_len, dh}
                                            k, // { nkvh,total_len, dh}
                                            v, // { nkvh,total_len, dh}
                                            attention_scale,
                                            head_dim,
                                            num_attention_heads,
                                            num_key_value_heads,
                                            ngroup,
                                            ntoken,
                                            total_token);

        return att_val;
    }

    return query_states;
}
} // namespace infinicore::op
