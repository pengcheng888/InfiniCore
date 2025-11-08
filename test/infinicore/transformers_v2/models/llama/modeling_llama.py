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

from typing import Callable, Optional, Union

from ...cache_utils import Cache, DynamicCache
from ...generation.utils_wpc import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_outputs_wpc import BaseModelOutputWithPast, CausalLMOutputWithPast

from ...modeling_utils_wpc import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from .configuration_llama import LlamaConfig

logger = logging.get_logger(__name__)

import infinicore
import torch
from torch import nn

LlamaRMSNorm = infinicore.nn.RMSNorm


class LlamaMLP(infinicore.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = infinicore.nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = infinicore.nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = infinicore.nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = infinicore.nn.functional.silu

    def forward(self, x: infinicore.Tensor) -> infinicore.Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs: Unpack[TransformersKwargs]):
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(infinicore.nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size

        self.rope_infinicore = infinicore.nn.RoPE(config)

        self.q_proj = infinicore.nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = infinicore.nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = infinicore.nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = infinicore.nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(self,
                hidden_states: infinicore.Tensor,
                attention_mask: Optional[torch.Tensor],
                past_key_values: Optional[Cache] = None,
                cache_position: Optional[torch.LongTensor] = None,
                **kwargs: Unpack[TransformersKwargs],
                ) -> tuple[Union[infinicore.Tensor, torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
 
        query_hidden_shape = (*input_shape, self.num_attention_heads, self.head_dim)
        key_hidden_shape = (*input_shape, self.num_key_value_heads, self.head_dim)
        value_hidden_shape = (*input_shape, self.num_key_value_heads, self.head_dim)

        query_states_infinicore = self.q_proj(hidden_states).view(query_hidden_shape)
        key_states_infinicore = self.k_proj(hidden_states).view(key_hidden_shape)
        value_states_infinicore = self.v_proj(hidden_states).view(value_hidden_shape).permute((0, 2, 1, 3))
      
        query_states = self.rope_infinicore.forward(query_states_infinicore, cache_position)
        key_states = self.rope_infinicore.forward(key_states_infinicore, cache_position)
 
        query_states = infinicore.convert_infini_to_torch_tensor(query_states).permute((0, 2, 1, 3)).contiguous()
        key_states = infinicore.convert_infini_to_torch_tensor(key_states).permute((0, 2, 1, 3)).contiguous()
        value_states = infinicore.convert_infini_to_torch_tensor(value_states_infinicore)
     
    
        # kv cache
        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"cache_position": cache_position}
    
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    
        if True:
            query_states_infini = infinicore.convert_torch_to_infini_tensor(query_states.contiguous())
            key_states_infini =   infinicore.convert_torch_to_infini_tensor(key_states.contiguous())
            value_states_infini = infinicore.convert_torch_to_infini_tensor(value_states.contiguous())
            
       
            att_val =  infinicore.attention_lm(query_states_infini,
                                    key_states_infini,
                                    value_states_infini,
                                    )
        
            bs = input_shape[0]
            ntok = input_shape[1]
            ngroup =  self.num_attention_heads // self.num_key_value_heads
            nkvh = self.num_key_value_heads
            dh = self.head_dim
            
    
            att_val_torch = infinicore.convert_infini_to_torch_tensor(att_val)
            att_val_torch = att_val_torch.reshape((bs, nkvh, ngroup, ntok, dh )).reshape((bs,self.num_attention_heads , ntok, dh )).contiguous()
            attn_output =  att_val_torch.permute((0, 2, 1,3)).contiguous() # ==> {bs, ntok, self.num_attention_heads , dh }
            
            # ([bs, ntok, num_attention_heads, head_dim]) 
            attn_output = attn_output.reshape(*input_shape, -1).contiguous() # ==> [bs, ntok, hidden_size] 
        


        if False:
            # attention
            attention_interface: Callable = eager_attention_forward
            if self.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            # Size([bs, ntok, num_attention_heads, head_dim])  Size([1, 5, 32, 64])
            attn_output, attn_weights = attention_interface(
                self,
                query_states,  # [bs, num_attention_heads, ntok, head_dim]
                key_states,  # [bs, num_key_value_heads, all_tok, head_dim]
                value_states,  # [bs, num_key_value_heads, all_tok, head_dim]
                attention_mask,  # [1, 1, ntok, all_tok]
                dropout=0.0,  #
                scaling=self.scaling,  # 缩放系数 0.125
                **kwargs,  # 'position_ids': tensor([[0, 1, 2, 3, 4]])
            )
            
        

            # temp = attn_output.permute((0, 2, 1, 3)).contiguous().reshape((1, 4, 40, 64))
            # print("attn_output: ",temp.shape)
            # print(temp)
            attn_output = attn_output.reshape(*input_shape, -1).contiguous() # ==> [bs, ntok, hidden_size] 
    
        else:
            #attention(query_states, key_states, value_states, k_cache, v_cache, pos, *, out=None)
            pass
        
        # o_proj    
        attn_output_infinicore = infinicore.convert_torch_to_infini_tensor(attn_output)
        attn_output = self.o_proj(attn_output_infinicore)

        return attn_output


class LlamaDecoderLayer(infinicore.nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self,
                hidden_states: infinicore.Tensor,  # [bs, ntok, hidden_size]
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Cache] = None,
                use_cache: Optional[bool] = False,
                cache_position: Optional[torch.LongTensor] = None,  # necessary, but kept here for BC
                **kwargs: Unpack[TransformersKwargs],
                ) -> torch.Tensor:
        
        residual = hidden_states
        # Self Attention
        hidden_states = self.input_layernorm(hidden_states)

 
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LlamaModel(infinicore.nn.Module):  # LlamaPreTrainedModel  torch.nn.Module
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
  
        self.embed_tokens = infinicore.nn.Embedding(config.vocab_size, config.hidden_size)
   
        self.layers = infinicore.nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
      
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None, # tensor([[    1,  1128,   526,   366, 29892]])
            attention_mask: Optional[torch.Tensor] = None, # torch.Size([1, 1, 5, 14])
            position_ids: Optional[torch.LongTensor] = None, # tensor([[0, 1, 2, 3, 4]])
            past_key_values: Optional[Cache] = None, # StaticCache(layers=[StaticLayer])
            inputs_embeds: Optional[torch.FloatTensor] = None, # None
            cache_position: Optional[torch.LongTensor] = None, # tensor([0, 1, 2, 3, 4])
            use_cache: Optional[bool] = None, # True
            **kwargs: Unpack[TransformersKwargs],# {}
    ) -> BaseModelOutputWithPast:
        
   
        
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            # input_ids :     {1,5}       tensor([[    1,  1128,   526,   366, 29892]])
            # inputs_embeds : {1,5,2048}  tensor([[[...]]])
            # input_ids = input_ids.to(dtype=torch.int32)
        
   
            input_ids_infini = infinicore.convert_torch_to_infini_tensor(input_ids.to(device="cpu"))
            inputs_embeds_infini = self.embed_tokens(input_ids_infini)
            inputs_embeds = infinicore.convert_infini_to_torch_tensor(inputs_embeds_infini)
            
        if use_cache and past_key_values is None: # 下面不执行
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None: # 下面不执行
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:  # 下面不执行
            position_ids = cache_position.unsqueeze(0)
   
        # ---------------------------------------------------------------------- #
        #            等完成attention接口后，下面的代码就不用了                        #
        # ---------------------------------------------------------------------- #
        causal_mask = create_causal_mask(config=self.config,
                                         input_embeds=inputs_embeds,
                                         attention_mask=attention_mask,
                                         cache_position=cache_position,
                                         past_key_values=past_key_values,
                                         position_ids=position_ids)
    
  
        hidden_states = inputs_embeds
        hidden_states = infinicore.convert_torch_to_infini_tensor(hidden_states)
      
        ilayer = 0
        for decoder_layer in self.layers[:self.config.num_hidden_layers]:
            print("ilayer: ", ilayer)
            ilayer += 1
         
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = infinicore.convert_infini_to_torch_tensor(hidden_states)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states,
                                       past_key_values=past_key_values,
                                       last_hidden_state_last_token=hidden_states[:, [-1], :])


class LlamaPreTrainedModel(PreTrainedModel):
    config: LlamaConfig
    base_model_prefix = "model"
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):  # torch.nn.Module LlamaPreTrainedModel,
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = infinicore.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @can_return_tuple
    @auto_docstring
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state  # [1,5,2048]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep  # [0,None,None]

        if outputs.last_hidden_state_last_token is not None:  # torch.Size([1, 2048])
            logits = self.lm_head(outputs.last_hidden_state_last_token)  # logits torch.Size([1, 1, 32000])
            return CausalLMOutputWithPast(
                logits=logits,
                next_token_logits=logits,
                past_key_values=outputs.past_key_values
            )
        else:
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            return CausalLMOutputWithPast(
                logits=logits,
                past_key_values=outputs.past_key_values
            )


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
]
