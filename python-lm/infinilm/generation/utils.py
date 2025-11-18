from typing import Optional

import infinicore

from ..cache_utils import Cache, DynamicCache


class GenerationMixin:
    def _get_initial_cache_position(self, seq_length, device):
        """Calculates `cache_position` for the pre-fill stage"""

        cache_position_list = list(range(0, seq_length))
        return infinicore.from_list(
            cache_position_list,
            dtype=infinicore.int64,
            device=device,
        )

    def prepare_inputs_for_generation(
        self,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ):
        """Prepare the model inputs for generation. It includes operations like slicing inputs given the existing cache."""

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
        #                 所需的: 其他
        # -------------------------------------------------------------------- #
        # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        return model_inputs

    def generate(self, input_ids, max_new_tokens=10, device=None, **kwargs):
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
            **model_kwargs,
        )
        return result

    def _sample(
        self,
        input_ids,
        max_new_tokens=10,
        device=None,
        **model_kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.

            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        """

        batch_size, cur_len = input_ids.shape[:2]

        # -------------------------------------------------------------------------- #
        #                     初始化 cache_position
        # -------------------------------------------------------------------------- #
        cache_position = self._get_initial_cache_position(cur_len, device)

        # model_forward = self.__call__
        cur_count = 0
        output_tokens_list = []

        model_kwargs["input_ids"] = input_ids
        model_kwargs["cache_position"] = cache_position
        while cur_count < max_new_tokens:
            # -------------------------------------------------------------------------- #
            #                     prepare model inputs
            # -------------------------------------------------------------------------- #
            model_inputs = self.prepare_inputs_for_generation(**model_kwargs)

            # -------------------------------------------------------------------------- #
            #                     计算一次
            # -------------------------------------------------------------------------- #
            outputs = self.forward(**model_inputs, return_dict=True)

            # -------------------------------------------------------------------------- #
            #                     更新下一次所需的，cache_position
            # -------------------------------------------------------------------------- #

            cache_position = model_kwargs["cache_position"]
            (seq_len,) = cache_position.shape  # [5] [1]

            last_position = cache_position.narrow(0, seq_len - 1, 1)
            one_value = infinicore.from_list(
                [1],
                dtype=last_position.dtype,
                device=last_position.device,
            )
            next_position = one_value + last_position

            model_kwargs["cache_position"] = next_position

            # -------------------------------------------------------------------------- #
            #                     处理输出
            # -------------------------------------------------------------------------- #
            next_token_scores = outputs.logits

            # -------------------------------------------------------------------------- #
            #                     random_sample
            # -------------------------------------------------------------------------- #
            # random_sample : token selection
            batch_size, _, vocab_size = next_token_scores.shape

            next_tokens = infinicore.empty(
                (batch_size,),
                dtype=infinicore.int32,
                device=next_token_scores.device,
            )

            for i in range(0, batch_size):
                score = next_token_scores.narrow(0, i, 1).view([vocab_size])

                out = next_tokens.narrow(0, i, 1).view([])
                infinicore.nn.functional.random_sample(
                    score,
                    1.0,
                    0.0,
                    1,
                    1.0,
                    out=out,
                )

            # ----------------------------------------------------------------- #
            #                        得到cpu上的结果
            # ----------------------------------------------------------------- #
            # update generated ids, model inputs, and length for next step
            next_tokens = next_tokens.to_torch().cpu()

            next_token = next_tokens[0].item()  # 将 torch 中的数据 转为 python的int类型
            output_tokens_list.append(next_token)
            model_kwargs["next_token"] = next_token

            cur_len += 1
            cur_count += 1

            del outputs

        return output_tokens_list
