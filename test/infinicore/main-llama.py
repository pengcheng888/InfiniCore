import os
import torch

import transformers_v2


def func(Folder):
    # ------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------ #

    from transformers_v2 import LlamaForCausalLM
    # model = LlamaForCausalLM.from_pretrained(Folder,
    #                                          dtype=torch.bfloat16, device_map="auto",
    #                                          attn_implementation="sdpa")

    model = LlamaForCausalLM.from_pretrained(Folder,
                                             dtype=torch.float16, device_map="cpu",
                                             attn_implementation="sdpa")

    # ------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------ #
    from transformers_v2 import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(Folder)

    # ------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------ #
    prompt = ["How are you,"]  # {'input_ids': tensor([[    1,  1128,   526,   366, 29892]]), 'attention_mask': tensor([[1, 1, 1, 1, 1]])}
    prompt = "How are you,"  # {'input_ids': tensor([[    1,  1128,   526,   366, 29892]]), 'attention_mask': tensor([[1, 1, 1, 1, 1]])}
    # prompt = ["How are you,",
    #           "How old are you,"]  # {'input_ids': tensor([[1,1128,526,366, 29892,2],  [1, 1128, 2030, 526, 366, 29892]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])}
    input_ids = tokenizer(prompt,
                          padding=True,  # 自动填充到相同长度
                          truncation=True,  # 自动截断到最大长度
                          max_length=128,  # 设置最大长度
                          return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**input_ids, cache_implementation="static", max_new_tokens=10)
        for output in outputs:
            print(tokenizer.decode(output, skip_special_tokens=True))


if __name__ == '__main__':
    Folder = r'/home/ubuntu/workspace_aisys/tensorRT_quantization-main/Llama/Llama2-TinyLlama-1.1B-Chat-v1.0/'
    Folder = r'/home/ubuntu/workspace_aisys/tensorRT_quantization-main/Llama/TinyLlama-1.1B-Chat-v1.0-small/'
    # Folder = r'/home/ubuntu/models/TinyLlama-1.1B-Chat-v1.0-small/'
    func(Folder)