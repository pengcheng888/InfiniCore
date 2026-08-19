# python deepseek_v4_silu_and_mul.py --hygon > deepseek_v4_silu_and_mul.log
# python deepseek_v4_rmsnorm_self.py --hygon > deepseek_v4_rmsnorm_self.log
# python deepseek_v4_mhc_pre.py --hygon > deepseek_v4_mhc_pre.log
# python deepseek_v4_mhc_post.py --hygon > deepseek_v4_mhc_post.log
# python deepseek_v4_hc_head.py --hygon > deepseek_v4_hc_head.log
# python deepseek_v4_mhc_fused_post_pre.py --hygon > deepseek_v4_mhc_fused_post_pre.log
# python deepseek_v4_embedding_and_hc_expand.py --hygon > deepseek_v4_embedding_and_hc_expand.log

# python deepseek_v4_biased_topk.py --hygon > deepseek_v4_biased_topk.log
# python deepseek_v4_hash_topk.py --hygon > deepseek_v4_hash_topk.log
# python deepseek_v4_linear_bf16_fp32.py --hygon > deepseek_v4_linear_bf16_fp32.log
# python deepseek_v4_topk_transform_512.py --hygon > deepseek_v4_topk_transform_512.log

python deepseek_v4_fused_rope.py --hygon > deepseek_v4_fused_rope.log
python deepseek_v4_fused_q_norm_rope.py --hygon > deepseek_v4_fused_q_norm_rope.log
python deepseek_v4_fused_k_norm_rope_flashmla.py --hygon > deepseek_v4_fused_k_norm_rope_flashmla.log

