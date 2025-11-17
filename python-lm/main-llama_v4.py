import infinilm
import torch
from infinilm.modeling_utils import get_model_state_dict

import infinicore


def func(Folder):
    # ---------------------------------------------------------------------------- #
    #                        创建 tokenizer
    # ---------------------------------------------------------------------------- #
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(Folder)

    # ---------------------------------------------------------------------------- #
    #                        创建模型, 加载权重
    # ---------------------------------------------------------------------------- #
    model_device = "cuda"
    infini_device = infinicore.device(model_device, 0)
    infini_dtype = infinicore.float32

    model = infinilm.LlamaForCausalLM.from_pretrained(
        model_path=Folder, device=infini_device, dtype=infini_dtype
    )
    model_param_infini = get_model_state_dict(model_path=Folder, device=model_device)

    assert (
        model_param_infini["model.embed_tokens.weight"].dtype == infinicore.float32
    ), "使用float32的权重类型"

    model.load_state_dict(model_param_infini)

    # ---------------------------------------------------------------------------- #
    #                        token编码
    # ---------------------------------------------------------------------------- #
    # prompt = ["How are you,"]
    prompt = "How are you,"
    # 'input_ids': tensor([[ 1, 1128, 526, 366, 29892]]
    input_ids = tokenizer(
        prompt,
        padding=True,  # 自动填充到相同长度
        truncation=True,  # 自动截断到最大长度
        max_length=128,  # 设置最大长度
        return_tensors="pt",
    )
    input_ids = input_ids.to(model_device)

    # ---------------------------------------------------------------------------- #
    #                        自回归生成
    # ---------------------------------------------------------------------------- #
    inputs_tensor = input_ids["input_ids"]
    input_ids_infini = inputs_tensor.cpu().to_infini()

    output_tokens_list = model.generate(
        input_ids_infini, max_new_tokens=10, device=infini_device
    )

    print(output_tokens_list)

    # ---------------------------------------------------------------------------- #
    #                        解码成字符显示
    # ---------------------------------------------------------------------------- #
    outputs = torch.tensor([output_tokens_list])
    print("prompt:\n", "How are you,")
    for output in outputs:
        print(tokenizer.decode(output, skip_special_tokens=True))
    print("\n\nover!")


if __name__ == "__main__":
    Folder = r"/home/ubuntu/workspace_aisys/tensorRT_quantization-main/Llama/Llama2-TinyLlama-1.1B-Chat-v1.0/"
    # Folder = r"/home/ubuntu/workspace_aisys/tensorRT_quantization-main/Llama/TinyLlama-1.1B-Chat-v1.0-small/"
    # Folder = r"/home/ubuntu/models/TinyLlama-1.1B-Chat-v1.0-small/"
    # Folder = r"/home/ubuntu/models/TinyLlama-1.1B-Chat-v1.0/"

    func(Folder)
