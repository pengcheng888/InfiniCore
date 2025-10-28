import os
import torch

import   transformers_v2



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
    prompt = "How are you,"
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(input_ids)

    output = model.generate(**input_ids, cache_implementation="static", max_new_tokens=5)
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == '__main__':
    Folder = r'/home/ubuntu/workspace_aisys/tensorRT_quantization-main/Llama/Llama2-TinyLlama-1.1B-Chat-v1.0/'
    # Folder = r'/home/ubuntu/models/TinyLlama-1.1B-Chat-v1.0/'
    func(Folder)