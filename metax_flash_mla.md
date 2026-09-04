
> 运行/workspace\_codex/InfiniLM/vllm\_offline\_basic.py脚本，分析用的是哪个flash mla的算子，导入的哪个包。

<details><summary>16 previous messages</summary>

> 我先看脚本内容和它引用的 vLLM/FlashMLA 入口，再实际运行脚本抓 import/调用路径。这个脚本可能会触发模型加载，所以我会先判断运行成本。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `source /.myenv.sh
> sed -n '1,260p' vllm_offline_basic.py`
> - Ran `source /.myenv.sh
> rg -n "flash_mla|flashmla|_flashmla|FlashMLA|vllm|LLM|deepseek_v4|mla" vllm_offline_basic.py /opt/conda/lib/python3.12/site-packages/vllm -S`
> - Ran `source /.myenv.sh
> ls -l vllm_offline_basic.py
> pwd`
>
> </details>
>
> 脚本本身只 `from vllm import LLM, SamplingParams`，创建的是 `/data-aisoft/mechdancer/models/GLM-4.7-Flash`、TP=4。现在我实际运行它，看初始化时选到哪个 MLA backend、是否真正导入 FlashMLA 包。
>
> <details><summary>Ran a command</summary>
>
> - Ran `source /.myenv.sh
> python vllm_offline_basic.py`
>
> </details>
>
> 脚本已经启动，vLLM 的 `metax` platform plugin 被激活了；这意味着后续不是标准 CUDA FlashMLA 路线。我继续等模型初始化日志，里面通常会打印 attention backend。
>
> 运行日志显示模型架构被 MetaX 插件重写成 `vllm_metax.models.deepseek_v4:DeepseekV4ForCausalLM` 一类路径，当前还在启动 worker。我继续等到它进入 forward 或报错，才能确认实际算子。
>
> 运行已经确认选择了 `FLASHMLA` backend，并且日志里出现 `Using Maca version of flash attention`。我同时去看 `vllm_metax` 插件代码，确定它最终导入的是哪个扩展包、调用哪个函数名。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `source /.myenv.sh
> python - <<'PY'
> import vllm_metax, inspect, os
> print(vllm_metax.__file__)
> PY`
> - Ran `source /.myenv.sh
> rg -n "FLASHMLA|flash_mla|flashmla|_flashmla|fwd_kvcache|dense_decode|get_mla|Maca version of flash attention" /opt/conda/lib/python3.12/site-packages/vllm_metax /opt/conda/lib/python3.12/site-packages/vllm -S`
> - Ran `source /.myenv.sh
> python - <<'PY'
> import sys
> mods=['vllm.v1.attention.ops.flashmla','vllm.third_party.flashmla.flash_mla_interface','flash_mla_cuda','vllm._flashmla_C','vllm._flashmla_extension_C']
> for m in mods:
>     try:
>         mod=__import__(m, fromlist=['*'])
>         print(m, 'OK', getattr(mod, '__file__', None))
>     except Exception as e:
>         print(m, 'ERR', type(e).__name__, e)
> PY`
>
> </details>
>
> Context compaction
>
> Source: automatic
> Status: completed
>
> 我先把正在跑的 `vllm_offline_basic.py` 收尾看完，然后补一眼 `flash_mla` 包本身的实现，这样能把“日志里选择的后端”和“实际导入的 Python/so 包”对应起来。
>
> 脚本已经跑通了，生成了输出；日志确认走的是 `FLASHMLA` 后端。现在我把 `vllm_metax` 和顶层 `flash_mla` 包的绑定关系对一下，避免只凭日志猜算子名。
>
> <details><summary>Explored a search, ran a command</summary>
>
> - Ran `source /.myenv.sh
> python - <<'PY'
> import flash_mla, inspect
> print('flash_mla file:', getattr(flash_mla, '__file__', None))
> print('flash_mla package attrs sample:', [x for x in dir(flash_mla) if 'mla' in x.lower() or 'flash' in x.lower()])
> import flash_mla.flash_mla_interface as i
> print('interface file:', i.__file__)
> print('flash_mla_cuda attr:', getattr(i, 'flash_mla_cuda', None))
> for name in ['flash_mla_with_kvcache','get_mla_metadata','flash_mla_sparse_fwd']:
>     obj = getattr(i, name, None)
>     print('\nFUNC', name, obj)
>     try:
>         print(inspect.getsource(obj))
>     except Exception as e:
>         print('source_error:', type(e).__name__, e)
> PY`
>
> <details><summary>Explored a search</summary>
>
> - Searched for `flash_mla_cuda|fwd_kvcache_mla|get_mla_metadata|dense_decode|sparse_prefill|flash_mla_with_kvcache|FLASHMLA` in `flash_mla`
>
> </details>
>
> </details>
>
> 这里已经能看到关键链路了：`vllm_metax` 不是去 import `vllm.third_party.flashmla.flash_mla_interface`，而是 import 顶层包 `flash_mla.flash_mla_interface`，里面再 `import flash_mla_cuda as flash_mla`。我再拿到具体文件行号和实际 so 路径。
>
> <details><summary>Explored 3 files, ran a command</summary>
>
> - Ran `source /.myenv.sh
> python - <<'PY'
> import flash_mla.flash_mla_interface as i
> print('interface:', i.__file__)
> print('native module object:', i.flash_mla)
> print('native module file:', getattr(i.flash_mla, '__file__', None))
> print('native exported names:', [n for n in dir(i.flash_mla) if not n.startswith('_')])
> for name in ['fwd_kvcache_mla','get_mla_metadata','sparse_prefill_fwd','fwd_kvcache_mla_fp8']:
>     obj = getattr(i.flash_mla, name, None)
>     print(name, obj)
>     print(getattr(obj, '__doc__', '') or '')
> PY`
>
> <details><summary>Explored 3 files</summary>
>
> - Read `/opt/conda/lib/python3.12/site-packages/vllm_metax/v1/attention/ops/flashmla.py`
> - Read `/opt/conda/lib/python3.12/site-packages/vllm_metax/v1/attention/backends/mla/flashmla.py`
> - Read `/opt/conda/lib/python3.12/site-packages/flash_mla/flash_mla_interface.py`
>
> </details>
>
> </details>
>
> 实际 native 模块已经确认：`flash_mla.flash_mla_interface` 绑定的是 `/opt/conda/lib/python3.12/site-packages/flash_mla_cuda.cpython-312-x86_64-linux-gnu.so`，导出的 decode 入口叫 `fwd_kvcache_mla`，不是上游 vLLM 的 `torch.ops._flashmla_C.dense_decode_fwd`。
>
> <details><summary>Explored 2 files, a search</summary>
>
> - Read `/workspace_codex/InfiniLM/vllm_offline_basic.py`
> - Read `/opt/conda/lib/python3.12/site-packages/vllm_metax/platform.py`
> - Searched for `Using FLASHMLA|Valid backends|Using FlashAttention prefill|attention backend|VLLM_ATTENTION_BACKEND|FLASHMLA` in `vllm_metax`
>
> </details>
</details>

已运行：

```bash
source /.myenv.sh
python /workspace_codex/InfiniLM/vllm_offline_basic.py
```

脚本运行成功，最终输出：

```text
Prompt: 'Hello, my name is'
Output: ' Matthew. I am 22 years old. I am currently in my second year'
```

**结论**

这个脚本本身只导入了：

[workspace_codex/InfiniLM/vllm_offline_basic.py](/workspace_codex/InfiniLM/vllm_offline_basic.py:4)

```python
from vllm import LLM, SamplingParams
```

运行时 MetaX 插件接管了 vLLM 的 attention backend。日志里明确显示：

```text
Valid backends: ['FLASHMLA', 'TRITON_MLA']
Using FLASHMLA attention backend out of potential backends: ('FLASHMLA', 'TRITON_MLA')
Using FlashAttention prefill for MLA
```

实际使用的 FlashMLA backend 是：

[vllm_metax/platform.py](/opt/conda/lib/python3.12/site-packages/vllm_metax/platform.py:83)

```python
vllm_metax.v1.attention.backends.mla.flashmla.MacaFlashMLABackend
```

FlashMLA 的 Python 包导入链路是：

[vllm_metax/v1/attention/ops/flashmla.py](/opt/conda/lib/python3.12/site-packages/vllm_metax/v1/attention/ops/flashmla.py:13)

```python
import flash_mla
from flash_mla.flash_mla_interface import flash_mla_with_kvcache, get_mla_metadata
```

而 `flash_mla.flash_mla_interface` 里面实际导入的是：

[flash_mla/flash_mla_interface.py](/opt/conda/lib/python3.12/site-packages/flash_mla/flash_mla_interface.py:5)

```python
import flash_mla_cuda as flash_mla
```

对应 native so 是：

```text
/opt/conda/lib/python3.12/site-packages/flash_mla_cuda.cpython-312-x86_64-linux-gnu.so
```

**实际调用的算子**

decode 阶段调用链是：

1. `MacaFlashMLABackend`
2. `FlashMLAImpl.forward_mqa`
3. `flash_mla_with_kvcache(...)`
4. `flash_mla_cuda.fwd_kvcache_mla(...)`

关键调用位置：

[vllm_metax/v1/attention/backends/mla/flashmla.py](/opt/conda/lib/python3.12/site-packages/vllm_metax/v1/attention/backends/mla/flashmla.py:317)

```python
o, lse = flash_mla_with_kvcache(...)
```

[flash_mla/flash_mla_interface.py](/opt/conda/lib/python3.12/site-packages/flash_mla/flash_mla_interface.py:220)

```python
out, softmax_lse = flash_mla.fwd_kvcache_mla(...)
```

所以这里用的不是 vLLM 上游的 `torch.ops._flashmla_C.dense_decode_fwd`，而是 MetaX wheel 里的：

```text
flash_mla_cuda.fwd_kvcache_mla
flash_mla_cuda.get_mla_metadata
```

其中 `get_mla_metadata` 用来生成 decode metadata，`fwd_kvcache_mla` 是实际的 FlashMLA dense decode/kvcache 前向算子。_prefill_ 日志显示走的是 FlashAttention prefill，不是 FlashMLA dense decode。

> flash\_mla\_with\_kvcache函数的cp\_world\_size，cp\_rank，cp\_tot\_seqlen\_k是什么意思。
>
> 当tp=4时，每个卡上调用时，数值分别是多少。

<details><summary>6 previous messages</summary>

> 我会在本地包和 InfiniCore 代码里查一下这几个参数的使用点，尤其区分 TP 和 CP；这三个名字看起来是 context parallel 相关，不能直接按 TP=4 推断。
>
> <details><summary>Explored a search, ran a command</summary>
>
> - Ran `source /.myenv.sh
> python - <<'PY'
> import flash_mla.flash_mla_interface as i
> print(i.flash_mla.fwd_kvcache_mla.__doc__)
> PY`
>
> <details><summary>Explored a search</summary>
>
> - Searched for `cp_world_size|cp_rank|cp_tot_seqlen|dcp_tot|context parallel|DCP|decode context` in `flash_mla`
>
> </details>
>
> </details>
>
> 搜索结果里有一条很有用：vLLM 把 `cp_world_size/cp_rank` 对应到了 DCP，也就是 decode context parallel。接下来我看一下 DCP 默认值和调用时是否真的传给 MetaX FlashMLA。
>
> <details><summary>Explored 4 files</summary>
>
> - Read `./config/parallel.py`
> - Read `./v1/attention/backends/mla/flashattn_mla.py`
> - Read `/opt/conda/lib/python3.12/site-packages/vllm_metax/model_executor/layers/attention/mla_attention.py`
> - Read `/opt/conda/lib/python3.12/site-packages/vllm_metax/v1/attention/backends/mla/flashmla.py`
>
> </details>
>
> 对 MetaX 的 `FLASHMLA` 路径有个关键发现：当前 `vllm_metax` 的 `flash_mla_with_kvcache` 调用没有显式传 `cp_world_size/cp_rank/cp_tot_seqlen_k`，所以会落到 `flash_mla` 包的默认值。也就是说 TP=4 不等于这三个参数就是 4/各 rank。
>
> <details><summary>Explored 3 files</summary>
>
> - Read `/opt/conda/lib/python3.12/site-packages/vllm_metax/v1/attention/backends/mla/common.py`
> - Read `./v1/attention/backend.py`
> - Read `./distributed/parallel_state.py`
>
> </details>
</details>

`flash_mla_with_kvcache` 里的这三个参数是 **CP/DCP 参数**，不是 TP 参数。

在当前环境的 MetaX FlashMLA 包里定义如下：

[flash_mla_interface.py](/opt/conda/lib/python3.12/site-packages/flash_mla/flash_mla_interface.py:147)

```python
def flash_mla_with_kvcache(...,
    cp_world_size=1,
    cp_rank=0,
    cp_tot_seqlen_k=None,
)
```

含义：

`cp_world_size`：Context Parallel / Decode Context Parallel 的并行组大小。表示 KV cache 的上下文长度是否被拆到多个 rank 上。

`cp_rank`：当前 rank 在 CP/DCP 组里的编号，不是 TP rank。

`cp_tot_seqlen_k`：DCP 场景下每个请求原始的完整 KV sequence length。因为 DCP 后每张卡可能只持有一部分 KV，所以 kernel 需要知道全局总长度来做 attention 语义修正。

结合 vLLM 代码看，DCP 默认是 1：

[parallel.py](./config/parallel.py:308)

```python
decode_context_parallel_size: int = 1
```

而你当前脚本只是：

[vllm_offline_basic.py](/workspace_codex/InfiniLM/vllm_offline_basic.py:20)

```python
LLM(..., tensor_parallel_size=4)
```

没有设置 `decode_context_parallel_size`，所以 **TP=4 不会让这三个参数变成 4**。

当前 `vllm_metax` 的 FlashMLA decode 调用也没有显式传这三个参数：

[vllm_metax/v1/attention/backends/mla/flashmla.py](/opt/conda/lib/python3.12/site-packages/vllm_metax/v1/attention/backends/mla/flashmla.py:317)

```python
o, lse = flash_mla_with_kvcache(...)
```

所以会使用默认值。

因此，当 `tp=4` 且没有启用 DCP 时，4 张卡上每次调用都是：

```text
TP rank 0: cp_world_size=1, cp_rank=0, cp_tot_seqlen_k=None
TP rank 1: cp_world_size=1, cp_rank=0, cp_tot_seqlen_k=None
TP rank 2: cp_world_size=1, cp_rank=0, cp_tot_seqlen_k=None
TP rank 3: cp_world_size=1, cp_rank=0, cp_tot_seqlen_k=None
```

只有当显式启用 `decode_context_parallel_size=4` 时，才会变成类似：

```text
DCP rank 0: cp_world_size=4, cp_rank=0, cp_tot_seqlen_k=<全局原始 seq_lens tensor>
DCP rank 1: cp_world_size=4, cp_rank=1, cp_tot_seqlen_k=<全局原始 seq_lens tensor>
DCP rank 2: cp_world_size=4, cp_rank=2, cp_tot_seqlen_k=<全局原始 seq_lens tensor>
DCP rank 3: cp_world_size=4, cp_rank=3, cp_tot_seqlen_k=<全局原始 seq_lens tensor>
```

一句话：**TP=4 时，当前脚本每卡仍然是 `cp_world_size=1, cp_rank=0, cp_tot_seqlen_k=None`；这三个参数只跟 DCP/CP 有关，不跟普通 TP 直接绑定。**

> 运行/workspace\_codex/InfiniLM/run\_infer.sh，处理报错。

<details><summary>25 previous messages</summary>

> 我来直接跑 `run_infer.sh` 复现错误，然后顺着 traceback 定位。会先带上 `source /.myenv.sh`，因为这个环境变量对 MetaX/vLLM 路径选择很关键。
>
> <details><summary>Ran a command</summary>
>
> - Ran `source /.myenv.sh
> bash run_infer.sh`
>
> </details>
>
> 报错已经出现：是在 InfiniCore 的 pinnable block allocator 里 `infinirtMalloc` 失败，底层 `mcMalloc` 返回 Error Code 2。先等脚本完全退出，再看 `run_infer.sh` 的启动参数和 allocator 相关配置。
>
> 脚本退出了，核心不是 Python 导入问题，而是设备内存分配失败。现在我看启动脚本、默认模型/TP/cache 参数，以及 InfiniLM 里 `num_blocks` 是怎么计算出来的。
>
> <details><summary>Explored a file, a search, ran a command</summary>
>
> <details><summary>Explored a file</summary>
>
> - Read `./run_infer.sh`
>
> </details>
>
> - Ran `git status --short`
>
> <details><summary>Explored a search</summary>
>
> - Searched for `num_blocks|block_size|Paged KV|enable_paged|paged|kv_cache|gpu_memory|memory|cache` in `python`
>
> </details>
>
> </details>
>
> `run_infer.sh` 没传 `--num-blocks`，所以 `examples/test_infer.py` 用默认 `512` 个 paged KV block；这会在 4 卡上预分配一大块 KV cache。下一步我先用更小的 `--num-blocks` 直接验证是不是 cache 规模导致的 `mcMalloc` 失败。
>
> <details><summary>Ran a command</summary>
>
> - Ran `source /.myenv.sh
> python examples/test_infer.py --device metax --model=/data-aisoft/mechdancer/models/GLM-4.7-Flash --temperature 1.0 --top-p 0.8 --top-k 10 --tp 4 --max-new-tokens 4 --block-size 64 --num-blocks 128 --warmup --prompt "山东最高的山是" --enable-paged-attn --attn flash-attn`
>
> </details>
>
> `num_blocks=128` 已经越过了之前立刻失败的位置，说明方向基本对：默认 cache 预分配太大。继续等它完成或出现下一处问题。
>
> 现在在加载 48 个 safetensors 分片，`num_blocks=128` 没再触发初始化 OOM。等加载和推理完成后，我会把脚本改成可配置的较小默认值，再用 `run_infer.sh` 本身验证。
>
> 分片加载到后半段了，目前没有新的异常。这个测试如果跑通，就可以把 `run_infer.sh` 的 `NUM_BLOCKS` 参数显式化，默认先设成 128，保留环境变量覆盖能力。
>
> `num_blocks=128` 已经完成了推理，但进程在退出阶段又被 glibc 报 `corrupted size vs. prev_size` 中止。也就是说有两个问题：初始化 OOM 已解决方向明确，剩下是退出清理阶段的 native 内存破坏；我会看 `test_infer.py` 和引擎 shutdown 逻辑，优先做最小修复。
>
> <details><summary>Explored a file, 2 searches</summary>
>
> - Read `./examples/test_infer.py`
> - Searched for `def shutdown|shutdown\(|__del__|atexit|join\(|stop\(|close\(|terminate|RankWorker|thread_loop|destroy` in `python`
> - Searched for `num_blocks|block-size|enable-paged-attn` in `test_infer.py`
>
> </details>
>
> 我发现一个更明显的 Python 侧问题：`examples/test_infer.py` 在正式生成前先调用了一次 `model.chat()`，并且在 `finally` 里把 `model.close()` 了，然后下面又继续用同一个 `model` 推理。这很容易造成底层 native 对象被提前销毁/重复清理，和退出阶段的内存破坏吻合。准备把这段重复生成删掉，并让脚本只在真正全部完成后关闭一次模型。
>
> <details><summary>Explored a file, ran a command</summary>
>
> <details><summary>Explored a file</summary>
>
> - Read `./python/infinilm/llm/llm.py`
>
> </details>
>
> - Ran `git diff -- examples/test_infer.py run_infer.sh python/infinilm/llm/llm.py`
>
> </details>
>
> 我会做两个很小的改动：`test_infer.py` 去掉提前关闭后复用模型的流程；`run_infer.sh` 显式传 `NUM_BLOCKS`，默认 128，后续可以用环境变量调大。这样同时覆盖初始化 OOM 和退出 abort 两个症状。
>
> <details><summary>Edited 2 files</summary>
>
> - Wrote `/workspace_codex/InfiniLM/examples/test_infer.py` (+25 -28)
> - Wrote `/workspace_codex/InfiniLM/run_infer.sh` (+2 -0)
>
> </details>
>
> 改完了。现在用 `run_infer.sh` 本身验证，确保它不再走默认 512 blocks，也不再提前 close 后继续推理。
>
> <details><summary>Ran a command</summary>
>
> - Ran `source /.myenv.sh
> bash run_infer.sh`
>
> </details>
>
> 验证脚本已经显示 `Using Paged KV Cache with num_blocks=128`，初始化 OOM 的那段已经过去。继续等权重加载和推理结束，看退出码是否干净。
>
> 权重还在加载中，速度正常。这里我不打断，让它完整跑完，主要观察最后是否还会 `Aborted`。
>
> 加载进入最后四分之一，仍然稳定。上一轮直接命令在推理结束后 abort，这次改过 `test_infer.py` 后重点看退出清理是否正常。
>
> `run_infer.sh` 现在完整跑完且退出码 0，推理输出正常。最后我看一下 diff，整理一下改动点和原因。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `git diff -- run_infer.sh examples/test_infer.py`
> - Ran `git status --short`
>
> </details>
>
> 有个小细节：`git diff` 里只显示了 `examples/test_infer.py`，`run_infer.sh` 没在 Git 状态里，说明它可能是未纳入版本控制或被忽略的本地脚本；但文件内容确实已更新并验证过。再确认一下脚本行号。
>
> <details><summary>Explored a file, a list</summary>
>
> - Read `./run_infer.sh`
> - Listed files in `run_infer.sh`
>
> </details>
</details>

已处理并验证 `run_infer.sh`，现在能完整跑完，退出码为 0。

修复了两个问题：

1. `run_infer.sh` 默认没有传 `--num-blocks`，`test_infer.py` 会使用默认 `512`，导致 MetaX 上 paged KV cache 预分配过大，触发：

```text
mcMalloc failed
pinnable_block_allocator.cc:77
```

我在 [run_infer.sh](/workspace_codex/InfiniLM/run_infer.sh:20) 里加了：

```bash
NUM_BLOCKS="${NUM_BLOCKS:-128}"
```

并传给脚本：

```bash
--num-blocks "$NUM_BLOCKS"
```

后续需要调大可以这样跑：

```bash
NUM_BLOCKS=256 bash run_infer.sh
```

2. [examples/test_infer.py](/workspace_codex/InfiniLM/examples/test_infer.py:150) 原来会先 `model.chat()`，然后 `finally: model.close()`，接着又继续使用同一个已经 close 的 `model` 生成。这会导致推理结束后 native 资源清理异常，之前复现到：

```text
corrupted size vs. prev_size while consolidating
Aborted (core dumped)
```

我改成只执行一次真正的生成，并在生成结束后 `model.close()` 一次。

验证结果：

```text
Using Paged KV Cache with num_blocks=128
...
Resquest 0:
===Response===
1. **分析
...
total_time: 1928.16 ms
```

`bash run_infer.sh` 已成功完成，没有再 abort。注意当前仓库里还有一些已有改动我没碰：`csrc/models/deepseek_v4/deepseek_v4_c4_indexer.cpp`、`xmake.lua`、`vllm_offline_basic.py`。

