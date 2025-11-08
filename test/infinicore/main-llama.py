import os
import torch
import numbers
import transformers_v2
import time




def func(Folder):
    # ------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------ #

    # model = LlamaForCausalLM.from_pretrained(Folder,
    #                                          dtype=torch.bfloat16, device_map="auto",
    #                                          attn_implementation="sdpa")

    model = transformers_v2.LlamaForCausalLM.from_pretrained(Folder,
                                             dtype=torch.float16, device_map="cuda",
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
    
    #prompt = "山东最高的山是"  # {'input_ids': tensor([[    1,  1128,   526,   366, 29892]]), 'attention_mask': tensor([[1, 1, 1, 1, 1]])}
    # prompt = ["How are you,",
    #           "How old are you,"]  # {'input_ids': tensor([[1,1128,526,366, 29892,2],  [1, 1128, 2030, 526, 366, 29892]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])}


    # 'input_ids': tensor([[ 1, 1128, 526, 366, 29892]]
    input_ids = tokenizer(prompt,
                          padding=True,  # 自动填充到相同长度
                          truncation=True,  # 自动截断到最大长度
                          max_length=128,  # 设置最大长度
                          return_tensors="pt").to(model.device)
   
    with torch.no_grad():
        print('------> start')
        t1 = time.time()
        outputs,_ = model.generate(**input_ids, max_new_tokens=15)  # cache_implementation="static",
        t2 = time.time()
        print("time: ", (t2 - t1) * 1000)
        
        for output in outputs:
            print(tokenizer.decode(output, skip_special_tokens=True))


if __name__ == '__main__':
    Folder = r'/home/ubuntu/workspace_aisys/tensorRT_quantization-main/Llama/Llama2-TinyLlama-1.1B-Chat-v1.0/'
    Folder = r'/home/ubuntu/workspace_aisys/tensorRT_quantization-main/Llama/TinyLlama-1.1B-Chat-v1.0-small/'
    # Folder = r'/home/ubuntu/models/TinyLlama-1.1B-Chat-v1.0-small/'
    func(Folder)