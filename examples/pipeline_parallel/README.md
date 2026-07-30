# 昇腾上的 PP×TP/EP 与 PP+权重卸载

本目录提供一条从 Qwen3-8B Dense 模型起步、再扩展到大规模 MoE 模型的可复现实验路径。

`parallel_inference.py` 会先校验模型类型、可见 NPU 数和并行拓扑，再启动 `vllm serve`。它不会修改 vLLM 的并行实现，只负责避免错误参数进入昂贵的模型加载阶段。

## 文件说明

| 文件 | 作用 |
|---|---|
| `parallel_inference.py` | PP×TP/EP、PP+权重卸载的统一启动与参数校验入口 |
| `launch_pp_serve.sh` | 兼容原有 `PP=2, TP=1` 调用方式的薄封装 |
| `bench_serve.py` | 无额外依赖的在线吞吐压测 |
| `find_idle_npu.sh` | 查找空闲 NPU die |
| `docker.sh` | 启停 vLLM-Ascend 开发容器 |
| `install_sources.sh` | 源码开发模式辅助脚本；本教程的独立镜像模式不使用 |
| `doc/qwen3-parallel-offload-report.md` | 实验报告、结果表和放大模型决策门 |

## 关键概念

- Dense Qwen3-8B 没有专家层，因此可验证 `PP×TP`，不能验证 EP。
- MoE 模型开启 `--enable-expert-parallel` 后，专家并行组由现有 TP/DP 组派生；当 `DP=1` 时，实验拓扑仍按 `PP×TP` 计算可见设备数。
- PP+权重卸载时，每个 PP rank 只卸载本 stage 持有的层。
- 昇腾优先使用 `prefetch` 权重卸载。它支持异步 H2D 预取和 ACLGraph。`uva` 在昇腾上会退化为 functional-call 搬运，只适合 eager 诊断。

### 为什么主实验没有 `--cpu-offload-gb`

vLLM 的这组参数对应两种可替代的权重卸载后端，不需要同时填写：

| 后端 | 关键参数 | 含义 | 本教程用途 |
|---|---|---|---|
| `prefetch` | `--offload-group-size`、`--offload-num-in-group`、`--offload-prefetch-step`、`--offload-params` | 按层和参数选择卸载，并在计算前异步预取回 NPU | 昇腾主实验 |
| `uva` | `--cpu-offload-gb N` | 按每个 NPU 的 GiB 预算卸载；该参数用于选择/配置 UVA 路径 | 仅 eager 诊断 |

因此，主实验使用 `--offload-backend prefetch` 时没有
`--cpu-offload-gb` 是正常且有意的；不要把两套参数混用。服务器镜像中的
`vLLM 0.25.1` 日志已经显示 `AscendPrefetchOffloader`，并报告每 rank
权重驻留下降及主机内存增加，证明执行的确是 CPU 权重卸载，而不只是命令被接受。

如果验收标准明确要求必须出现 `--cpu-offload-gb`，应把下文的 UVA eager
用例作为单独补充项，不能用它替换已经完成的 prefetch 显存与性能对照。

## 0. 固定镜像、容器与模型目录

本轮实验选择服务器已有的最新官方 A3 镜像：

```text
quay.io/ascend/vllm-ascend:nightly-main-a3
```

`jzc/vllm-ascend-pp:latest` 的创建时间更新一天，但它是自定义镜像，版本和源码基线不透明；首轮可复现实验优先使用官方 nightly。自定义镜像可在基线通过后作为对照组。

截图对应的本地镜像 ID 为 `a7788a65d91b`。实验前记录实际 ID：

```bash
docker image inspect quay.io/ascend/vllm-ascend:nightly-main-a3 \
  --format '{{.Id}} {{json .RepoDigests}}'
```

把本目录完整复制到服务器：

```bash
scp -r pipeline_parallel root@服务器地址:/home/vllm/l00977701/
```

在服务器上启动容器：

```bash
cd /home/vllm/l00977701/pipeline_parallel
chmod +x *.sh

export IMAGE=quay.io/ascend/vllm-ascend:nightly-main-a3
export CONTAINER_NAME=qwen3_parallel_nightly

./docker.sh start
./docker.sh check
```

如果目录曾由 Windows 工具转换过换行符，并出现
`/bin/bash^M: bad interpreter`，先在服务器执行：

```bash
sed -i 's/\r$//' *.sh
chmod +x *.sh
```

脚本会根据当前位置自动生成以下映射：

- `/home/vllm/l00977701/pipeline_parallel` → `/workspace/pipeline_parallel`；
- `/home/vllm/l00977701/runtime` → `/workspace`；
- `/home/vllm/l00977701/models` → `/models`。

vLLM 和 vLLM-Ascend 直接使用官方镜像内已经安装的版本，不需要复制两个源码仓库，也不要运行 `install_sources.sh`。`check` 通过后再进入容器：

```bash
./docker.sh shell
```

### 下载 Qwen3-8B

只在 `/models/Qwen3-8B/config.json` 不存在时下载：

```bash
if [[ ! -s /models/Qwen3-8B/config.json ]]; then
  if command -v ms-hub >/dev/null 2>&1; then
    ms-hub download Qwen/Qwen3-8B \
      --local-dir /models/Qwen3-8B \
      --max-workers 8
  elif command -v modelscope >/dev/null 2>&1; then
    modelscope download \
      --model Qwen/Qwen3-8B \
      --local_dir /models/Qwen3-8B
  else
    python3 - <<'PY'
from modelscope import snapshot_download

snapshot_download("Qwen/Qwen3-8B", local_dir="/models/Qwen3-8B")
PY
  fi
fi
```

`/models` 默认映射到宿主机 `/home/vllm/l00977701/models`。删除容器或切换镜像不会删除权重。

下载后校验：

```bash
python3 - <<'PY'
import json
from pathlib import Path

model_dir = Path("/models/Qwen3-8B")
config = json.loads((model_dir / "config.json").read_text())
weight_files = sorted(model_dir.glob("*.safetensors"))
assert weight_files, "No safetensors weights found"
print("architectures:", config.get("architectures"))
print("model_type:", config.get("model_type"))
print("num_hidden_layers:", config.get("num_hidden_layers"))
print("weight_shards:", len(weight_files))
PY

du -sh /models/Qwen3-8B
```

## 1. 启动前检查

```bash
cd /workspace/pipeline_parallel

python3 parallel_inference.py \
  --model /models/Qwen3-8B \
  --devices 0,1,2,3 \
  --pp 2 \
  --tp 2 \
  --dry-run
```

输出包含最终环境变量和逐 token 的 `vllm serve` 参数。设备数必须满足：

```text
visible_devices = PP × TP
```

## 2. Qwen3-8B：先跑通 PP

先用两个 NPU 验证 PP 数据流。首次定位问题时可添加 `--enforce-eager`，成功后去掉它验证 ACLGraph。

```bash
python3 /workspace/pipeline_parallel/parallel_inference.py \
  --model /models/Qwen3-8B \
  --served-model-name qwen3-8b \
  --devices 0,1 \
  --pp 2 \
  --tp 1 \
  --max-model-len 4096 \
  --max-num-seqs 16
```

## 3. Qwen3-8B：PP×TP 二维并行

```bash
python3 /workspace/pipeline_parallel/parallel_inference.py \
  --model /models/Qwen3-8B \
  --served-model-name qwen3-8b \
  --devices 0,1,2,3 \
  --pp 2 \
  --tp 2 \
  --max-model-len 4096 \
  --max-num-seqs 16
```

四个 worker 的 rank 日志都必须出现，且服务请求必须返回非空文本。只看到 `Application startup complete` 不算通过。

## 4. Qwen3-8B：PP+prefetch 权重卸载

以下配置每四个本地 Transformer 层卸载一个，并只选择 MLP 的两个大矩阵。它是保守起点，不是固定最优值。

```bash
python3 /workspace/pipeline_parallel/parallel_inference.py \
  --model /models/Qwen3-8B \
  --served-model-name qwen3-8b-offload \
  --devices 0,1 \
  --pp 2 \
  --tp 1 \
  --offload-backend prefetch \
  --offload-group-size 4 \
  --offload-num-in-group 1 \
  --offload-prefetch-step 1 \
  --offload-params gate_up_proj,down_proj \
  --max-model-len 4096 \
  --max-num-seqs 16
```

日志中应出现：

```text
[NPUPrefetchOffloader] Initialized ...
Total NPU memory saved: ... GB
Static buffer pool: ... GB
```

先比较“PP 基线”和“PP+卸载”的 greedy 输出，再比较显存、TTFT、TPOT 和吞吐。不要仅以模型成功加载作为通过依据。

### 固定 KV cache 后测量显存

默认 `--gpu-memory-utilization` 会把权重卸载释放的空间重新分给 KV cache，
导致 `npu-smi` 总 HBM 不能直接反映卸载收益。显存 A/B 对照必须在基线和
卸载组中同时加入相同的固定值，例如每 NPU 16 GiB：

```bash
--kv-cache-memory-bytes 17179869184
```

启动器在设置此参数后不会再生成 `--gpu-memory-utilization`。其余 PP、TP、
最大长度和并发参数必须保持完全一致。

本次 Qwen3-8B 实测结果（ACLGraph、PP=2、TP=1、16 GiB KV/NPU）：

| 指标 | 无卸载 | prefetch | 变化 |
|---|---:|---:|---:|
| 每 rank 加载权重 | 7.6576 GB | 6.5326 GB | -1.1250 GB |
| 平均 HBM | 27889 MiB | 27031.5 MiB | -857.5 MiB |
| 平均 worker 进程显存 | 24944 MiB | 24085 MiB | -859 MiB |
| 容器主机内存 | 5.369 GiB | 9.259 GiB | +3.890 GiB |

这证明权重卸载确实降低 NPU 驻留，但当前 `group_size=4` 配置在短请求负载下
性能损失较大，只能作为功能验证起点，不能直接用于生产。

## 5. MoE 模型：PP×TP/EP

Qwen3-8B 通过后，建议用 Qwen3-30B-A3B 做第一条 MoE 验证链：

```bash
python3 /workspace/pipeline_parallel/parallel_inference.py \
  --model /models/Qwen3-30B-A3B \
  --served-model-name qwen3-30b-a3b \
  --devices 0,1,2,3 \
  --pp 2 \
  --tp 2 \
  --enable-ep \
  --max-model-len 4096 \
  --max-num-seqs 16
```

本地 `config.json` 必须包含有效专家数量。对 Dense 权重开启 `--enable-ep` 会在启动前被拒绝。

## 6. 功能验证

```bash
curl -sf http://127.0.0.1:8000/v1/models

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-8b",
    "messages": [{"role": "user", "content": "用一句话介绍流水线并行。"}],
    "temperature": 0,
    "max_tokens": 64
  }'
```

通过条件：

1. `/v1/models` 返回 HTTP 200。
2. chat 请求返回 HTTP 200，`choices[0].message.content` 非空。
3. 所有 rank 无 fatal、OOM、HCCL 超时或 missing weight。
4. 同一 greedy 请求与单卡/无卸载基线输出一致，或 logprob 差异在既定阈值内。

## 7. 性能压测

```bash
python3 /workspace/pipeline_parallel/bench_serve.py \
  --url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --num-prompts 64 \
  --concurrency 16 \
  --in-len 256 \
  --out-len 256
```

至少记录：

- 启动峰值 NPU/CPU 内存；
- 稳态每 rank NPU 内存；
- TTFT、TPOT、端到端时延；
- input/output token throughput；
- PP stage 空闲比例和 H2D 拷贝占比。

## 8. UVA 回退，仅用于诊断

```bash
python3 /workspace/pipeline_parallel/parallel_inference.py \
  --model /models/Qwen3-8B \
  --devices 0,1 \
  --pp 2 \
  --tp 1 \
  --offload-backend uva \
  --cpu-offload-gb 2 \
  --enforce-eager
```

该模式会自动设置 `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1`，使用 functional-call 回退。昇腾没有 CUDA UVA 零拷贝能力，不应把这个结果当作推荐性能路径。

## 9. 调参顺序

1. `PP=2, TP=1, offload=none, eager`：验证基本正确性。
2. 去掉 eager：验证 ACLGraph。
3. `PP=2, TP=2`：验证二维通信。
4. `PP=2, TP=1, prefetch`：验证权重卸载和显存收益。
5. `PP=2, TP=2, prefetch`：验证组合路径。
6. 换 MoE 权重并开启 EP。
7. 在准确性、稳定性和显存门槛通过后，再增加模型规模或上下文长度。

不要同时改变模型、并行度、图模式、卸载比例和并发；否则故障无法归因。

## 10. 本次真权重签收摘要

| 实验 | 结果 |
|---|---|
| Qwen3-8B 单卡 A0 | 16/16 短压测通过，251.3 output tok/s |
| Qwen3-8B PP×TP A2 | PP=2、TP=2；30 分钟 7072/7072 请求通过 |
| Qwen3-8B PP+prefetch B1 | 固定 KV 后平均降低 857.5 MiB/die HBM；30 分钟 1184/1184 请求通过 |
| Qwen3-30B-A3B PP×EP C1 | PP=2、TP=2、EP on；短测 49.7 output tok/s；30 分钟 1360/1360 请求通过 |

功能目标已经签收。当前 prefetch 配置的显存收益成立，但短请求吞吐损失仍较高，
因此适合作为功能验证基线，不应在未调优和未按业务负载复测前直接用于生产。
