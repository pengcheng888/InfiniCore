# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from typing import Optional, Union

from transformers.utils import logging

import infinicore

from ...cache_utils import Cache, DynamicCache
from ...generation.utils import GenerationMixin
from ...modeling_outputs import CausalLMOutputWithPast
from .configuration_llama import LlamaConfig

logger = logging.get_logger(__name__)

LlamaRMSNorm = infinicore.nn.RMSNorm


class LlamaMLP(infinicore.nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = infinicore.nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias, **kwargs
        )
        self.up_proj = infinicore.nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias, **kwargs
        )
        self.down_proj = infinicore.nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias, **kwargs
        )
        self.act_fn = infinicore.nn.functional.silu

    def forward(self, x: infinicore.Tensor) -> infinicore.Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class LlamaAttention(infinicore.nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int, **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size

        self.q_proj = infinicore.nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )

        self.k_proj = infinicore.nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )

        self.v_proj = infinicore.nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )

        self.o_proj = infinicore.nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: infinicore.Tensor,
        past_key_values: Optional[Cache] = None,
        rope_instance: infinicore.nn.RoPE = None,
        **kwargs,
    ) -> infinicore.Tensor:
        hidden_states_shape = hidden_states.shape  # [bs, seq_len, hidden_size]
        bs, seq_len = hidden_states_shape[:-1]  # [bs, seq_len]

        query_hidden_shape = (bs, seq_len, self.num_attention_heads, self.head_dim)
        key_hidden_shape = (bs, seq_len, self.num_key_value_heads, self.head_dim)
        value_hidden_shape = (bs, seq_len, self.num_key_value_heads, self.head_dim)

        # --------------------------------------------------------------------------------------- #
        #                           对 Q,K，V进行 project 加上 rope
        # --------------------------------------------------------------------------------------- #
        # => [bs, seq_len,  num_attention_heads, head_dim]
        query_states_infinicore = self.q_proj(hidden_states).view(query_hidden_shape)

        # => [bs, seq_len,  num_key_value_heads, head_dim]
        key_states_infinicore = self.k_proj(hidden_states).view(key_hidden_shape)
        # => [bs, seq_len, nkvh, head_dim]
        value_states_infinicore = self.v_proj(hidden_states).view(value_hidden_shape)

        # --------------------------------------------------------------------------------------- #
        #                           对 Q和K， 加上 rope
        # --------------------------------------------------------------------------------------- #
        cache_position = kwargs.pop("cache_position", None)
        if not cache_position:
            raise KeyError("cache_position error")

        if rope_instance is None:
            raise KeyError("rope_instance error")

        query_states = rope_instance(query_states_infinicore, cache_position)
        key_states = rope_instance(key_states_infinicore, cache_position)

        # --------------------------------------------------------------------------------------- #
        #                           kv cache
        # --------------------------------------------------------------------------------------- #

        # kv cache
        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {}
            key_states_infini, value_states_infini = past_key_values.update(
                key_states,  # [bs, seq_len, num_key_value_heads, head_dim]
                value_states_infinicore,  # [bs, seq_len, num_key_value_heads, head_dim]
                self.layer_idx,
                cache_kwargs,
            )

        # --------------------------------------------------------------------------------------- #
        #                           注意力计算
        # --------------------------------------------------------------------------------------- #

        if True:
            #  [bs, num_key_value_heads, seq_len, head_dim]
            query_states_infini = query_states.permute((0, 2, 1, 3)).contiguous()
            key_states_infini = key_states_infini.permute((0, 2, 1, 3)).contiguous()
            value_states_infini = value_states_infini.permute((0, 2, 1, 3)).contiguous()

            # att_val => [bs,  num_attention_heads, seq_len, head_dim]
            att_val = infinicore.nn.functional.self_attention(
                query_states_infini,  # [bs, num_attention_heads, seq_len, head_dim]
                key_states_infini,  # [bs, num_key_value_heads, all_tok, head_dim]
                value_states_infini,  # [bs, num_key_value_heads, all_tok, head_dim]
            )

            # => [bs, seq_len, num_attention_heads, dh ]
            attn_output = att_val.permute((0, 2, 1, 3)).contiguous()
        else:
            query_states_infini = query_states.contiguous()
            key_states_infini = key_states_infini.contiguous()
            value_states_infini = value_states_infini.contiguous()

            # att_val => [bs,  num_attention_heads, seq_len, head_dim]
            att_val = infinicore.nn.functional.self_attention(
                query_states_infini,  # [bs, num_attention_heads, seq_len, head_dim]
                key_states_infini,  # [bs, num_key_value_heads, all_tok, head_dim]
                value_states_infini,  # [bs, num_key_value_heads, all_tok, head_dim]
                is_causal=True,
                enable_gqa=True,
            )

            # => [bs, seq_len, num_attention_heads, dh ]
            attn_output = att_val.contiguous()

        # --------------------------------------------------------------------------------------- #
        #                           out project
        # --------------------------------------------------------------------------------------- #
        # ([bs, seq_len, num_attention_heads, head_dim]) ==> [bs, seq_len, hidden_size ]
        attn_output = attn_output.view(hidden_states_shape)

        # o_proj
        attn_output = self.o_proj(attn_output)

        return attn_output


class LlamaDecoderLayer(infinicore.nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int, **kwargs):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx, **kwargs)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: infinicore.Tensor,  # [bs, seq_len, hidden_size]
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        rope_instance=None,
        **kwargs,
    ) -> infinicore.Tensor:
        residual = hidden_states
        # Self Attention
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            rope_instance=rope_instance,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LlamaModel(infinicore.nn.Module):
    def __init__(self, config: LlamaConfig, **kwargs):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        self.embed_tokens = infinicore.nn.Embedding(
            config.vocab_size, config.hidden_size
        )

        self.layers = infinicore.nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx, **kwargs)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rope_instance = infinicore.nn.RoPE(
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            head_dim=self.head_dim,
            **kwargs,
        )

    def forward(
        self,
        input_ids,
        cache_position,
        past_key_values: Optional[Cache] = None,  # StaticCache(layers=[StaticLayer])
        use_cache: Optional[bool] = None,  # True
        **kwargs,  # {}
    ):
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # --------------------------------------------------------- #
        #               token的embedding
        # --------------------------------------------------------- #
        # input_ids :     {1,5}       tensor([[    1,  1128,   526,   366, 29892]])
        # inputs_embeds : {1,5,2048}  tensor([[[...]]])
        # input_ids = input_ids.to(dtype=int32)

        inputs_embeds = self.embed_tokens(input_ids)

        # --------------------------------------------------------- #
        #                    decoder_layer
        # --------------------------------------------------------- #
        ilayer = 0  # noqa: F841
        hidden_states = inputs_embeds
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            # print("ilayer: ", ilayer)
            # ilayer += 1

            hidden_states = decoder_layer(
                hidden_states,
                past_key_values=past_key_values,
                cache_position=cache_position,
                rope_instance=self.rope_instance,
                **kwargs,
            )

        # --------------------------------------------------------- #
        #                    norm
        # --------------------------------------------------------- #
        _, seq_len, _ = hidden_states.shape  # 1,5,2048
        last_token = hidden_states.narrow(1, seq_len - 1, 1)

        return self.norm(last_token)


class LlamaForCausalLM(infinicore.nn.Module, GenerationMixin):
    config: LlamaConfig

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config, **kwargs)
        self.vocab_size = config.vocab_size
        self.lm_head = infinicore.nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

    def forward(
        self,
        input_ids,
        cache_position,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        input_ids: Optional[ LongTensor ] = None,  # tensor([[    1,  1128,   526,   366, 29892]])
        cache_position: Optional[ LongTensor ] = None,  # [0,1,2,3,4] ...  [5]   cuda:0
        """

        last_token = self.model(
            input_ids,
            cache_position,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        # logits Size([1, 1, 32000])
        logits = self.lm_head(last_token)

        return CausalLMOutputWithPast(logits=logits)

    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[Union[str, os.PathLike]],
        device=None,
        dtype=None,
    ):
        def load_config_json(dir_path_: str):
            with open(os.path.join(dir_path_, "config.json"), "r") as f:
                config = json.load(f)
            return config

        config_dict = load_config_json(os.path.join(model_path))
        config = LlamaConfig(**config_dict)

        return LlamaForCausalLM(config, device=device, dtype=dtype)


__all__ = [
    "LlamaModel",
    "LlamaForCausalLM",
]
