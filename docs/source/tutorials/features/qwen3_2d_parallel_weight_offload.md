# Qwen3 推理二维并行与权重卸载

本教程在昇腾 NPU 上分阶段验证两种部署能力：

1. Pipeline Parallel 与 Tensor/Expert Parallel 组合的二维并行；
2. Pipeline Parallel 与 CPU 权重卸载组合。

建议先使用 Qwen3-8B Dense 权重跑通 PP×TP 和 PP+权重卸载，再使用 Qwen3-30B-A3B 等 MoE 权重验证 PP×EP，最后增加模型规模。

## 1. 支持边界

| 能力 | Qwen3-8B Dense | Qwen3 MoE |
|---|---:|---:|
| PP×TP | 支持 | 支持 |
| PP×EP | 不适用 | 支持 |
| PP+prefetch 权重卸载 | 支持 | 支持，需实测专家权重策略 |
| PP+UVA 权重卸载 | functional-call 回退 | functional-call 回退 |

EP 不是独立于 TP 的第三个 world-size 维度。开启 EP 后，专家组由 TP/DP 组派生；`DP=1` 时，可见设备数仍为 `PP × TP`。

## 2. 环境准备

本轮选择服务器已有的最新官方 A3 镜像：

```text
quay.io/ascend/vllm-ascend:nightly-main-a3
```

虽然服务器上的 `jzc/vllm-ascend-pp:latest` 创建时间更新，但它是自定义镜像，内部源码版本未知。为保证实验可复现，首轮选择最新官方 A3 nightly；自定义镜像只作为后续对照。

2026-07-27 截图中的本地镜像 ID 为 `a7788a65d91b`。开始实验前保存实际镜像信息：

```bash
docker image inspect quay.io/ascend/vllm-ascend:nightly-main-a3 \
  --format '{{.Id}} {{json .RepoDigests}}'
```

容器应至少挂载：

- 所需的 `/dev/davinci*`；
- `/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc`；
- Ascend driver、`npu-smi`；
- 模型目录与独立实验目录。

先把 `examples/pipeline_parallel` 整个目录复制到服务器的
`/home/vllm/l00977701/pipeline_parallel`，然后使用目录内的容器脚本：

```bash
cd /home/vllm/l00977701/pipeline_parallel
chmod +x *.sh

export IMAGE=quay.io/ascend/vllm-ascend:nightly-main-a3
export CONTAINER_NAME=qwen3_parallel_nightly

./docker.sh start
./docker.sh check
```

默认映射如下：

- 实验目录：`/home/vllm/l00977701/pipeline_parallel` → `/workspace/pipeline_parallel`；
- 运行目录：`/home/vllm/l00977701/runtime` → `/workspace`；
- 持久模型目录：`/home/vllm/l00977701/models` → `/models`。

vLLM 与 vLLM-Ascend 使用镜像内已安装的软件包。`check` 通过后执行
`./docker.sh shell` 进入容器。

进入容器后确认环境：

```bash
cd /workspace/pipeline_parallel

python3 - <<'PY'
import vllm
import vllm_ascend

print("vllm:", vllm.__file__)
print("vllm_ascend:", vllm_ascend.__file__)
PY
```

### 2.1 下载持久化权重

新镜像不包含 Qwen3-8B 权重。只在模型不存在时下载：

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

`/models` 默认映射到宿主机 `/home/vllm/l00977701/models`，因此重建容器或切换镜像不会重新下载。

检查权重：

```bash
MODEL=/models/Qwen3-8B
test -f "$MODEL/config.json"
grep -E '"architectures"|"model_type"|"num_hidden_layers"|"num_experts"' "$MODEL/config.json"
find "$MODEL" -maxdepth 1 -name '*.safetensors' -type f | sort
du -sh "$MODEL"
```

## 3. 使用统一启动器

本文使用：

```text
examples/pipeline_parallel/parallel_inference.py
```

它在启动前校验：

- 可见 NPU 数是否等于 `PP × TP`；
- EP 是否误用于 Dense 模型；
- prefetch 分组参数是否合法；
- 昇腾 UVA 回退是否启用 eager。

先执行 dry-run：

```bash
cd /workspace/pipeline_parallel

python3 /workspace/pipeline_parallel/parallel_inference.py \
  --model /models/Qwen3-8B \
  --devices 0,1,2,3 \
  --pp 2 \
  --tp 2 \
  --dry-run
```

## 4. Qwen3-8B PP×TP

### 4.1 PP=2 基线

```bash
python3 /workspace/pipeline_parallel/parallel_inference.py \
  --model /models/Qwen3-8B \
  --served-model-name qwen3-8b \
  --devices 0,1 \
  --pp 2 \
  --tp 1 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --enforce-eager
```

eager 路径通过后去掉 `--enforce-eager`，验证 ACLGraph 路径。

### 4.2 PP=2, TP=2

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

设备顺序决定 rank 映射。优先把同一 TP group 放在高速互联范围内；跨机 PP 时必须验证 RoCE/HCCL 网络。

## 5. Qwen3-8B PP+权重卸载

昇腾推荐 prefetch 后端：

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

参数含义：

- `offload_group_size=4`：每四个本 stage 层为一组；
- `offload_num_in_group=1`：每组卸载最后一层；
- `offload_prefetch_step=1`：提前一个卸载层发起 H2D；
- `offload_params`：只卸载匹配的 MLP 参数段。

`offload_prefetch_step` 越大，越可能隐藏 H2D 时延，但静态 NPU buffer 也越大。最终显存收益应以每个 rank 的初始化日志和 `npu-smi` 为准。

### 5.1 固定 KV cache 的显存对照

如果使用默认 `--gpu-memory-utilization`，vLLM 会把卸载释放的空间重新分配给
KV cache。此时权重驻留虽然下降，`npu-smi` 总 HBM 却可能不降。

严格的 A/B 对照应让无卸载组和卸载组都固定相同 KV cache，例如：

```bash
--kv-cache-memory-bytes 17179869184
```

该值表示每 NPU 固定 16 GiB KV cache。统一启动器在设置该参数后会省略
`--gpu-memory-utilization`，避免两套内存策略同时出现。两组实验还必须保持
PP、TP、最大长度、最大并发、eager/ACLGraph 模式及请求负载一致。

Qwen3-8B 实测（ACLGraph、PP=2、TP=1）：

| 指标 | 无卸载 | prefetch | 变化 |
|---|---:|---:|---:|
| 每 rank 加载权重 | 7.6576 GB | 6.5326 GB | -1.1250 GB |
| KV cache | 16.00 GiB | 16.00 GiB | 0 |
| 平均 HBM | 27889 MiB | 27031.5 MiB | -857.5 MiB |
| 平均 worker 进程显存 | 24944 MiB | 24085 MiB | -859 MiB |
| 容器主机内存 | 5.369 GiB | 9.259 GiB | +3.890 GiB |

因此固定 KV 是显存对照的必要条件。本配置证明卸载功能有效，但短请求压测中
性能损失较大；扩大模型前应降低卸载比例并继续调优预取距离。

## 6. MoE 模型 PP×EP

Qwen3-8B 是 Dense 模型，不能用于 EP 验收。换用本地 Qwen3-30B-A3B：

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

更大 MoE 模型沿用相同方法，但必须重新评估 PP 层切分、专家负载均衡、每 rank 权重体积和通信拓扑。

## 7. 请求与正确性验收

```bash
curl -sf http://127.0.0.1:8000/v1/models

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-8b",
    "messages": [{"role": "user", "content": "计算 23×17，只输出结果。"}],
    "temperature": 0,
    "max_tokens": 32
  }'
```

每个配置至少完成：

1. 真权重加载，无 missing/unexpected weight 致命错误；
2. `/v1/models` 返回 200；
3. 首个 chat 请求返回 200 和非空文本；
4. 固定 prompt、seed、greedy 参数与无卸载基线比较；
5. 连续运行 30 分钟，无 worker 退出、HCCL 超时和持续内存增长。

## 8. 性能与容量测试

```bash
python3 /workspace/pipeline_parallel/bench_serve.py \
  --url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --num-prompts 64 \
  --concurrency 16 \
  --in-len 256 \
  --out-len 256
```

对每个配置记录：

| 指标 | PP 基线 | PP×TP | PP+offload |
|---|---:|---:|---:|
| 每 rank 稳态 NPU 内存 | 待实测 | 待实测 | 待实测 |
| 主机内存 | 待实测 | 待实测 | 待实测 |
| TTFT P50/P95 | 待实测 | 待实测 | 待实测 |
| TPOT P50/P95 | 待实测 | 待实测 | 待实测 |
| output tokens/s | 待实测 | 待实测 | 待实测 |
| 30 分钟稳定性 | 待实测 | 待实测 | 待实测 |

只有准确性、稳定性、显存收益三个门槛都通过后，才增加模型规模或上下文长度。

## 9. 常见故障

- `Visible device count must match ...`：`--devices` 数量不等于 `PP×TP`。
- Dense 模型被拒绝 EP：换用 MoE 权重，不要绕过校验。
- `EI0015` 或 rank 连接超时：检查残留进程、设备映射、HCCL 超时和网络。
- 首请求崩溃：这是 false-ready，不能把 API 进程启动视为成功。
- 卸载后吞吐明显下降：减小卸载比例，增加预取步数前先确认 NPU buffer 余量，并分析 H2D 时间。
- ACLGraph 失败：先用 `--enforce-eager` 定位，再回到图模式完成最终验收。

完整实验矩阵和报告模板见
`examples/pipeline_parallel/doc/qwen3-parallel-offload-report.md`。
