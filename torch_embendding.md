> torch.nn.functional.embedding内部是如何实现的，调用的是哪个so的算子么

<details><summary>8 previous messages</summary>

> 我会继续使用 `hygon-bw1000-infinilm-deepseek-v4` skill，因为这个问题是在 SGLang/DeepSeek V4 的 embedding 调用链里追到底层算子。接下来我会看本机 PyTorch 的 `F.embedding` Python 源码和已加载的 Torch 动态库。
>
> <details><summary>Explored a file</summary>
>
> - Read `/root/.codex/skills/hygon-bw1000-infinilm-deepseek-v4-0727/SKILL.md`
>
> </details>
>
> 我先不改代码，只做本机追踪。`F.embedding` 通常只是 Python 包装层，真正执行会落到 `aten::embedding`，再由 PyTorch dispatcher 按设备选择 CPU/CUDA/HIP 后端。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `source ~/.bashrc && source /.myenv.sh && python - <<'PY'
> import torch, inspect
> import torch.nn.functional as F
> print('torch version:', torch.__version__)
> print('torch file:', torch.__file__)
> print('torch._C file:', torch._C.__file__)
> print('cuda available:', torch.cuda.is_available())
> print('hip version:', getattr(torch.version, 'hip', None))
> print('cuda version:', getattr(torch.version, 'cuda', None))
> print('F.embedding source file:', inspect.getsourcefile(F.embedding))
> print('F.embedding first line:', inspect.getsourcelines(F.embedding)[1])
> print(inspect.getsource(F.embedding))
> PY`
> - Ran `source ~/.bashrc && source /.myenv.sh && python - <<'PY'
> import torch, os, glob
> root=os.path.dirname(torch.__file__)
> print('torch root:', root)
> for p in sorted(glob.glob(os.path.join(root,'lib','*.so*'))):
>     print(p)
> PY`
>
> </details>
>
> 本机 Torch 是 `2.9.0`，HIP 版，`torch._C` 在 Python 扩展 `.so` 里，核心 ATen 库包括 `libtorch_cpu.so` 和 `libtorch_hip.so`。我再看 dispatcher 注册，确认 `aten::embedding` 本身是后端 kernel 还是复合实现。
>
> <details><summary>Ran a command</summary>
>
> - Ran `source ~/.bashrc && source /.myenv.sh && python - <<'PY'
> import torch
> for op in ['aten::embedding', 'aten::index_select', 'aten::embedding_renorm_']:
>     print('\n====', op, '====')
>     print(torch._C._dispatch_dump_table(op))
> PY`
>
> </details>
>
> dispatcher 表显示得很明确：`aten::embedding` 在 CPU/CUDA/HIP 上都是 `CompositeExplicitAutograd` 默认实现，本身不是一个单独的 HIP/CUDA 专用 embedding kernel；真正的数据搬运通常会继续落到 `aten::index_select` 这类后端 kernel。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `source ~/.bashrc && source /.myenv.sh && python - <<'PY'
> import torch
> print(torch.ops.aten.embedding.default)
> print(torch.ops.aten.index_select.default)
> PY`
> - Ran `nm -D /usr/local/lib/python3.10/dist-packages/torch/lib/libtorch_hip.so | c++filt | rg "index_select|embedding" | head -80`
> - Ran `nm -D /usr/local/lib/python3.10/dist-packages/torch/lib/libtorch_cpu.so | c++filt | rg "index_select|embedding" | head -80`
>
> </details>
</details>

`torch.nn.functional.embedding` 本身不是一个独立的 SGLang/InfiniCore 算子，它是 PyTorch 的 Python 包装，最终走 **ATen dispatcher**。

本机 Torch 版本是：

```text
torch 2.9.0
HIP 6.3.26113
```

调用链是：

```text
F.embedding(input, weight)
  -> torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
  -> aten::embedding
  -> aten::index_select
  -> 后端 index_select kernel
```

在 `/usr/local/lib/python3.10/dist-packages/torch/nn/functional.py` 里，`F.embedding` 最后就是：

```python
return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
```

从 dispatcher 看，本机的 `aten::embedding` 注册为：

```text
CompositeExplicitAutograd
```

也就是说，`aten::embedding` 本身不是一个单独的 HIP/CUDA 专用 kernel，而是 PyTorch C++ 里的组合实现。forward 的核心逻辑可以理解为：

```text
output = weight.index_select(0, input.reshape(-1)).view(input.shape + [hidden])
```

真正负责按 index 从 embedding table 里 gather 数据的是：

```text
aten::index_select
```

**对应 .so**

本机动态库里可以看到：

```text
/usr/local/lib/python3.10/dist-packages/torch/lib/libtorch_cpu.so
/usr/local/lib/python3.10/dist-packages/torch/lib/libtorch_hip.so
/usr/local/lib/python3.10/dist-packages/torch/_C.cpython-310-x86_64-linux-gnu.so
```

其中：

- Python 入口和 `torch.embedding` 绑定来自 `torch._C...so`
- `aten::embedding` 的 composite 实现在 `libtorch_cpu.so`
- 如果 tensor 在 HIP/GPU 上，实际数据搬运会落到 `libtorch_hip.so` 里的 `index_select` kernel

本机符号里能看到类似：

```text
libtorch_cpu.so:
  at::compositeexplicitautograd::embedding(...)
  at::cpu::index_select(...)

libtorch_hip.so:
  at::native::index_select_cuda(...)
```

这里 HIP 版 PyTorch 符号里仍然常用 `cuda` 命名，这是 PyTorch 历史命名，不代表它真的走 NVIDIA CUDA。

**对 VocabParallelEmbedding(enable_tp=False) 来说**

它实际走的是：

```text
VocabParallelEmbedding.forward
  -> _embed_local_shard
  -> UnquantizedEmbeddingMethod.embedding
  -> F.embedding
  -> torch.embedding
  -> aten::embedding
  -> aten::index_select
```

所以不是 SGLang 的 `fused_vocab_parallel_embedding` Triton kernel，也不是某个 InfiniCore 算子；它是 PyTorch ATen 的 embedding/index_select 路径。
