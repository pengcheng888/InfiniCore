import sys

import infinilm
import torch
from infinilm.modeling_utils import get_model_state_dict

import infinicore


def test(model_path, device_type="cuda"):
    # ---------------------------------------------------------------------------- #
    #                        创建 tokenizer
    # ---------------------------------------------------------------------------- #
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # ---------------------------------------------------------------------------- #
    #                        创建模型, 加载权重
    # ---------------------------------------------------------------------------- #
    infini_device = infinicore.device(device_type, 0)
    infini_dtype = infinicore.float16

    model = infinilm.LlamaForCausalLM.from_pretrained(
        model_path=model_path, device=infini_device, dtype=infini_dtype
    )
    model_param_infini = get_model_state_dict(
        model_path=model_path, infini_device=infini_device, infini_dtype=infini_dtype
    )

    assert (
        model_param_infini["model.embed_tokens.weight"].dtype == infinicore.float16
    ), " 使用支持 float16 的权重类型"

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
    input_ids = input_ids.to(device_type)

    # ---------------------------------------------------------------------------- #
    #                        自回归生成
    # ---------------------------------------------------------------------------- #
    inputs_tensor = input_ids["input_ids"]
    input_ids_infini = inputs_tensor.cpu().to_infini()

    import time

    t1 = time.time()
    output_tokens_list = model.generate(
        input_ids_infini, max_new_tokens=10, device=infini_device
    )
    t2 = time.time()
    print("time: ", (t2 - t1) * 1000)
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
    model_path = r"/home/ubuntu/workspace_aisys/tensorRT_quantization-main/Llama/Llama2-TinyLlama-1.1B-Chat-v1.0/"
    model_path = r"/home/ubuntu/workspace_aisys/tensorRT_quantization-main/Llama/TinyLlama-1.1B-Chat-v1.0-small/"
    model_path = r"/home/ubuntu/models/TinyLlama-1.1B-Chat-v1.0-small/"
    # model_path = r"/home/ubuntu/models/TinyLlama-1.1B-Chat-v1.0/"

    device_type = "cpu"
    test(model_path, device_type)
    exit(-1)

    if len(sys.argv) < 1:
        print("Usage: python main-llama.py [--cpu | --nvidia] <path/to/model_dir> ")
        sys.exit(1)

    # Parse command line arguments
    model_path = sys.argv[2]

    device_type = "cpu"
    if sys.argv[1] == "--cpu":
        device_type = "cpu"
    elif sys.argv[1] == "--nvidia":
        device_type = "cuda"
    else:
        print("Usage:  python main-llama.py [--cpu | --nvidia] <path/to/model_dir>")
        sys.exit(1)

    test(model_path, device_type)
