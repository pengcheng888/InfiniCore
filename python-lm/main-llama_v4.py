import sys
import time

import infinilm
from infinilm.modeling_utils import get_model_state_dict
from tokenizers import decoders as _dec
from transformers import AutoTokenizer

import infinicore


def test(model_path, device_str="cuda", max_new_tokens=100):
    # ---------------------------------------------------------------------------- #
    #                        创建模型,
    # ---------------------------------------------------------------------------- #
    infini_device = infinicore.device(device_str, 0)
    infini_dtype = infinicore.float16

    model = infinilm.LlamaForCausalLM.from_pretrained(
        model_path,
        device=infini_device,
        dtype=infini_dtype,
    )

    # ---------------------------------------------------------------------------- #
    #                        加载权重
    # ---------------------------------------------------------------------------- #
    model_param_infini = get_model_state_dict(
        model_path,
        device=infini_device,
        dtype=infini_dtype,
    )

    assert (
        model_param_infini["model.embed_tokens.weight"].dtype == infinicore.float16
    ), " 使用支持 float16 的权重类型"

    model.load_state_dict(model_param_infini)

    config = model.config

    # ---------------------------------------------------------------------------- #
    #                        创建 tokenizer
    # ---------------------------------------------------------------------------- #

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if "llama" == config.model_type:
        backend = getattr(tokenizer, "backend_tokenizer", None)
        target = getattr(backend, "_tokenizer", backend)
        norm = getattr(target, "normalizer", None)
        dec = getattr(target, "decoder", None)
        sn = repr(norm)[:800] if norm is not None else ""
        sd = repr(dec)[:800] if dec is not None else ""
        has_prepend = "Prepend" in sn
        has_strip = "Strip" in sd
        if has_prepend and has_strip:
            target.decoder = _dec.Sequence(
                [
                    _dec.Replace("▁", " "),
                    _dec.ByteFallback(),
                    _dec.Fuse(),
                ]
            )

    # ---------------------------------------------------------------------------- #
    #                        token编码
    # ---------------------------------------------------------------------------- #
    prompt = "How are you,"
    # prompt = "山东最高的山是？"
    if True:
        input_ids = tokenizer(
            prompt,
            padding=True,  # 自动填充到相同长度
            truncation=True,  # 自动截断到最大长度
            max_length=128,  # 设置最大长度
        )
        print("prompt: ", prompt)
        input_ids = input_ids["input_ids"]
    else:
        input_content = tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        print(input_content, end="", flush=True)
        input_ids = tokenizer.encode(input_content)

    # ---------------------------------------------------------------------------- #
    #                        自回归生成
    # ---------------------------------------------------------------------------- #

    input_ids_list = [input_ids]  # List: [[1, 1128, 526, 366, 29892]]
    input_ids_infini = infinicore.from_list(input_ids_list)

    t1 = time.time()
    output_tokens_list, output_content = model.generate(
        input_ids_infini,
        max_new_tokens=max_new_tokens,
        device=infini_device,
        tokenizer=tokenizer,
        config=config,
    )
    t2 = time.time()

    print(
        f"total_time: {(t2 - t1) * 1000} ms",
    )

    return


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="run Llama args")

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run CPU test",
    )
    parser.add_argument(
        "--nvidia",
        action="store_true",
        help="Run CPU test",
    )
    parser.add_argument(
        "--metax",
        action="store_true",
        help="Run CPU test",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型文件夹的路径",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="模型文件夹的路径",
    )

    return parser.parse_args()


if __name__ == "__main__":
    if True:
        model_path = r"/home/ubuntu/workspace_aisys/tensorRT_quantization-main/Llama/Llama2-TinyLlama-1.1B-Chat-v1.0/"
        model_path = r"/home/ubuntu/workspace_aisys/tensorRT_quantization-main/Llama/TinyLlama-1.1B-Chat-v1.0-small/"
        # model_path = r"/home/ubuntu/models/TinyLlama-1.1B-Chat-v1.0-small/"
        # model_path = r"/home/ubuntu/models/TinyLlama-1.1B-Chat-v1.0/"

        # model_path = r"/data-aisoft/mechdancer/models/TinyLlama-1.1B-Chat-v1.0/"

        device_type = "cuda"
        max_new_tokens = 10
        test(model_path, device_type, max_new_tokens=max_new_tokens)
        exit(-1)

    args = get_args()
    print(args)

    # Parse command line arguments
    # python python-lm/main-llama_v4.py --metax --model_path /data-aisoft/mechdancer/models/TinyLlama-1.1B-Chat-v1.0/
    device_type = "cpu"
    if args.cpu:
        device_type = "cpu"
    elif args.nvidia:
        device_type = "cuda"
    elif args.metax:
        device_type = "cuda"
    else:
        print(
            "Usage:  python examples/llama.py [--cpu | --nvidia] --model_path=<path/to/model_dir>"
        )
        sys.exit(1)

    model_path = args.model_path
    max_new_tokens = args.max_new_tokens

    test(model_path, device_type, max_new_tokens)
