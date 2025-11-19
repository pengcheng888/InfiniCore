from typing import Optional

import infinicore

from ..cache_utils import Cache, DynamicCache


class GenerationMixin:
    def _get_initial_cache_position(
        self,
        seq_length: int,
        device: infinicore.device,
    ) -> infinicore.Tensor:
        """Calculates `cache_position` for the pre-fill stage"""
        cache_position_list = list(range(0, seq_length))
        return infinicore.from_list(
            cache_position_list, dtype=infinicore.int64, device=device
        )

    def prepare_inputs_for_generation(
        self,
        device: infinicore.device,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ):
        """Prepare the model inputs for generation."""

        # 1. Handle BC:
        model_inputs = {}
        # -------------------------------------------------------------------- #
        #                 所需的: KV Cache
        # -------------------------------------------------------------------- #
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values

        # -------------------------------------------------------------------- #
        #                 所需的: cache_position
        # -------------------------------------------------------------------- #
        model_inputs["cache_position"] = kwargs.get("cache_position", None)

        # -------------------------------------------------------------------- #
        #                 所需的: token的input_ids
        # -------------------------------------------------------------------- #
        if kwargs.get("next_token", None) is not None:
            next_token = kwargs.get("next_token", None)
            input_ids = infinicore.from_list([[next_token]])
            model_inputs["input_ids"] = input_ids

        # -------------------------------------------------------------------- #
        #                 其他
        # -------------------------------------------------------------------- #
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        return model_inputs

    def generate(
        self,
        input_ids: infinicore.Tensor,
        max_new_tokens: int,
        device: infinicore.device,
        tokenizer,
        config,
        **kwargs,
    ):
        model_kwargs = kwargs

        # -------------------------------------------------------------------- #
        #                       创建 cache                                      #
        # -------------------------------------------------------------------- #
        model_kwargs["use_cache"] = True
        model_kwargs["past_key_values"] = DynamicCache(config=self.config)

        # -------------------------------------------------------------------- #
        #                       _sample函数                                     #
        # -------------------------------------------------------------------- #
        result = self._sample(
            input_ids,
            max_new_tokens=max_new_tokens,
            device=device,
            tokenizer=tokenizer,
            config=config,
            **model_kwargs,
        )
        return result

    def _sample(
        self,
        input_ids: infinicore.Tensor,
        max_new_tokens: int,
        device: infinicore.device,
        tokenizer,
        config,
        **model_kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head.

        Parameters:
            input_ids (batch_size, seq_len): The sequence used as a prompt for the generation.
            max_new_tokens: Maximum number of new tokens.
            device: infinicore.device.
            tokenizer: translating data into raw text.
        """

        batch_size, seq_len = input_ids.shape[:2]

        eos_token_id = config.eos_token_id
        eos_token_id_list = (
            [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
        )

        # -------------------------------------------------------------------------- #
        #                     初始化 cache_position
        # -------------------------------------------------------------------------- #

        output_tokens_list = []

        model_kwargs["input_ids"] = input_ids
        model_kwargs["cache_position"] = None
        output_content = ""
        print()

        for i in range(0, max_new_tokens):
            # -------------------------------------------------------------------------- #
            #                     计算所需的，cache_position
            # -------------------------------------------------------------------------- #
            current_cache_position = model_kwargs.get("cache_position", None)
            if current_cache_position is None:
                # prill阶段
                model_kwargs["cache_position"] = self._get_initial_cache_position(
                    seq_len, device
                )

            else:
                # decoder 阶段
                (seq_len,) = current_cache_position.shape
                last_position = current_cache_position.narrow(0, seq_len - 1, 1)

                one_value = infinicore.from_list(
                    [1],
                    dtype=last_position.dtype,
                    device=last_position.device,
                )
                next_position = one_value + last_position

                model_kwargs["cache_position"] = next_position

            # -------------------------------------------------------------------------- #
            #                     prepare model inputs
            # -------------------------------------------------------------------------- #
            model_inputs = self.prepare_inputs_for_generation(device, **model_kwargs)

            # -------------------------------------------------------------------------- #
            #                     计算一次
            # -------------------------------------------------------------------------- #
            logits = self.forward(**model_inputs, return_dict=True)

            # -------------------------------------------------------------------------- #
            #                     处理输出
            # -------------------------------------------------------------------------- #
            token_scores = logits

            # -------------------------------------------------------------------------- #
            #                     random_sample
            # -------------------------------------------------------------------------- #
            batch_size, _, vocab_size = token_scores.shape

            next_tokens = infinicore.empty(
                (batch_size,),
                dtype=infinicore.int32,
                device=token_scores.device,
            )
            for i in range(0, batch_size):
                score = token_scores.narrow(0, i, 1).view([vocab_size])
                out = next_tokens.narrow(0, i, 1).view([])
                infinicore.nn.functional.random_sample(
                    score,
                    0.8,
                    0.1,
                    1,
                    1.0,
                    out=out,
                )

            # ----------------------------------------------------------------- #
            #                得到下一个token的id，并解码为字符
            # ----------------------------------------------------------------- #
            token_id = next_tokens.to_numpy()[0]
            output_str = tokenizer.decode([token_id], skip_special_tokens=True)

            model_kwargs["next_token"] = token_id
            output_tokens_list.append(token_id)
            output_content += output_str

            print(output_str, end="", flush=True)
            if token_id in eos_token_id_list:
                break

        print()
        return output_tokens_list, output_content
