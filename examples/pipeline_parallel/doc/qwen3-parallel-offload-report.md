# Qwen3 昇腾二维并行与权重卸载实验报告

实验日期：2026-07-27 至 2026-07-28

代码分支：`pipeline_parallel`

实测软件基线：vLLM `0.25.1`，vLLM-Ascend nightly（包未提供版本字符串）

目标模型：Qwen3-8B Dense BF16

实验镜像：`quay.io/ascend/vllm-ascend:nightly-main-a3`

截图镜像 ID：`a7788a65d91b`

报告状态：Qwen3-8B 的 PP×TP、PP+prefetch，以及 Qwen3-30B-A3B 的 PP×EP 功能与 30 分钟稳定性验证均完成

镜像选择说明：`jzc/vllm-ascend-pp:latest` 创建时间更近，但属于自定义镜像且源码基线未知。首轮使用最新官方 A3 nightly 建立可复现基线，自定义镜像留作后续对照。

## 1. 结论摘要

当前代码基线已经具备完成任务所需的框架能力：

- PP 与 TP 可直接通过 `pipeline_parallel_size` 和 `tensor_parallel_size` 组合；
- MoE 模型可在相同 PP×TP world size 上开启 EP；
- 昇腾已有 prefetch CPU 权重卸载实现，可在 eager 和 ACLGraph 路径工作；
- PP rank 只加载本 stage 的层，因此权重卸载按 stage 本地生效。

本次补充的主要交付不是修改模型执行内核，而是：

1. 统一并校验 PP×TP/EP、PP+权重卸载启动参数；
2. 明确 Dense Qwen3-8B 与 MoE 模型的验收边界；
3. 增加 PP 基线与 PP+prefetch 真权重直接对比的两卡 E2E；
4. 形成中文教程、实验矩阵和放大模型的 go/no-go 门槛。

目标服务器已经完成容器、Python 包和 NPU 算子自检。Qwen3-8B 的 PP×TP
与 PP+prefetch 均完成真权重首请求和短压测；固定 KV 对照证明 prefetch
平均降低 857.5 MiB/die HBM，但当前卸载配置性能损失未达到生产门槛。

## 2. 前提与待确认项

当前按以下假设编写：

- “Qwen 8B”是 Qwen3-8B，而不是 Qwen2/2.5 的其他变体；
- 权重为可被当前 vLLM 直接加载的 BF16/FP16 或已支持量化格式；
- 首轮使用单机 Atlas A2/A3，至少 4 个可见 NPU；
- 实验目录位于宿主机 `/home/vllm/l00977701/pipeline_parallel`，容器内为 `/workspace/pipeline_parallel`；
- vLLM 与 vLLM-Ascend 使用官方镜像内安装的软件包；
- 容器运行目录为 `/workspace`；
- 模型位于 `/models/Qwen3-8B`。

实测前填写：

| 项目 | 实际值 |
|---|---|
| 模型绝对路径 | `/models/Qwen3-8B` |
| `config.json` 的 `architectures` | `Qwen3ForCausalLM` |
| 权重 dtype/量化 | BF16，未发现量化配置 |
| 权重文件 | 5 个 safetensors 分片，总目录约 16 GiB |
| 模型结构 | Dense；36 层，32 attention heads，8 KV heads，无专家层 |
| 服务器型号 | 待填写 |
| NPU 型号与数量 | Ascend 910；截图确认至少 5 个 die 可见，完整数量待记录 |
| 单 die 显存 | 65536 MiB |
| 容器镜像 | `quay.io/ascend/vllm-ascend:nightly-main-a3` |
| 容器镜像 ID/digest | `a7788a65d91b`，实测前重新记录 |
| CANN 版本 | 待填写 |
| torch/torch-npu 版本 | torch `2.10.0+cpu` / torch-npu `2.10.0.post2` |
| vLLM/vLLM-Ascend commit | vLLM `0.25.1`；vLLM-Ascend 版本字符串未提供，commit 待填写 |
| NPU 拓扑与 RoCE 网卡 | 待填写 |

## 3. 架构说明

### 3.1 PP×TP

当 `PP=2, TP=2, DP=1` 时，总 worker 数为 4：

```text
PP stage 0: TP rank 0, TP rank 1
PP stage 1: TP rank 0, TP rank 1
```

每个 PP stage 持有一段 Transformer 层；stage 内的张量权重再按 TP 切分。PP 降低单 rank 的层数，TP 降低单层权重和计算量。

### 3.2 PP×EP

EP 只适用于 MoE 专家层。vLLM 中 `--enable-expert-parallel` 使用现有 TP/DP 组派生专家组，不需要额外增加一个独立的 EP world-size 参数。

因此：

- Qwen3-8B Dense：验证 PP×TP，EP 为 N/A；
- Qwen3-30B-A3B：可使用 `PP=2, TP=2, EP=on`；
- 当 `DP=1` 时，可见设备数仍是 `PP × TP`。

### 3.3 PP+权重卸载

推荐 prefetch 后端。每个 PP rank：

1. 只遍历本 stage 的本地层；
2. 按 `offload_group_size` 分组；
3. 卸载每组最后 `offload_num_in_group` 个层的选定参数；
4. 用独立 NPU stream 提前搬回静态 buffer；
5. 当前计算 stream 等待对应事件后执行该层。

显存收益不是“卸载权重字节数”的简单相加，因为静态 prefetch buffer 会占用部分 NPU 内存。必须同时记录初始化日志和稳态 `npu-smi`。

`--cpu-offload-gb` 不是所有权重卸载方式都必须提供的通用开关。它配置的是
vLLM 的 UVA/容量预算路径；本次实验显式选择的是 `--offload-backend prefetch`，
由 `offload_group_size`、`offload_num_in_group`、`offload_prefetch_step`
和 `offload_params` 决定卸载范围与预取节奏。两套后端是替代关系，不应混用。

在当前昇腾环境中，prefetch 是主验证路径，UVA 会退化为 eager
functional-call 搬运，只作为兼容性诊断。因此本次 B1 命令没有
`--cpu-offload-gb` 属于预期行为。若项目验收口径指定该参数本身，则需增加一项
PP=2、`--offload-backend uva --cpu-offload-gb N --enforce-eager` 的独立补充
实验，但不能将其结果与 prefetch 的固定 KV 对照混为一组。

## 4. 代码证据审计

| 能力 | 代码/测试证据 | 审计结论 |
|---|---|---|
| PP=2 | `tests/e2e/pull_request/four_card/test_pipeline_parallel.py` | 已有 PP 测试，覆盖 mp/ray |
| PP×TP×EP | `tests/e2e/pull_request/four_card/test_deepseek_v3_2_w8a8_pruning.py` | 已有 `TP=2, PP=2, EP=on` 真权重用例 |
| Qwen3-8B 支持 | `tests/e2e/models/configs/Qwen3-8B.yaml` | 模型已有准确率配置 |
| NPU prefetch offloader | `vllm_ascend/model_executor/offloader/prefetch.py` | NPU stream/event 与静态 buffer 实现存在 |
| offloader 注册 | `vllm_ascend/worker/model_runner_v1.py` | vLLM prefetch offloader 被替换为 NPU 实现 |
| 单卡卸载路径 | `tests/e2e/pull_request/one_card/test_cpu_weight_offload.py` | 覆盖 eager/ACLGraph/选择性参数，但原比较 helper 不足以独立证明无卸载等价 |
| PP+卸载直接对比 | `tests/e2e/pull_request/two_card/test_pipeline_parallel_weight_offload.py` | 本次新增，待两卡 NPU 执行 |
| 参数编排 | `examples/pipeline_parallel/parallel_inference.py` | 本次新增，已完成 14 个 CPU 单测 |

已有分支文档还记录过 Qwen3-30B-A3B 的 `PP=2, TP=1` 运行，但它不能替代本次 Qwen3-8B 的真权重验收。

## 5. 最小实验矩阵

所有实验固定：

- 同一模型快照；
- `temperature=0`；
- 同一 prompt、seed、输入/输出长度；
- 先 eager 隔离，再验证 ACLGraph；
- 每次只改变一个变量。

| ID | 模型 | 拓扑 | 卸载 | 目的 | 状态 |
|---|---|---|---|---|---|
| A0 | Qwen3-8B | PP1×TP1 | 无 | 单 rank 精度基线 | 真权重启动、首请求与 16 请求短压测通过 |
| A1 | Qwen3-8B | PP2×TP1 | 无 | PP 基础链路 | 真权重启动、首请求与 16 请求短压测通过 |
| A2 | Qwen3-8B | PP2×TP2 | 无 | Dense 二维并行 | eager 启动、首请求与 16 请求短压测通过 |
| B1 | Qwen3-8B | PP2×TP1 | prefetch | PP+权重卸载 | 真权重启动、首请求与 16 请求短压测通过 |
| B2 | Qwen3-8B | PP2×TP2 | prefetch | 二维并行+卸载组合 | 待实测 |
| C1 | Qwen3-30B-A3B | PP2×TP2, EP on | 无 | PP×EP | 真权重、首请求、短压测及 30 分钟稳定性通过 |
| C2 | 更大 MoE | 按内存规划 | 可选 | 继续放大模型 | 尚未执行 |

### A2 启动命令

```bash
cd /workspace/pipeline_parallel

python3 /workspace/pipeline_parallel/parallel_inference.py \
  --model /models/Qwen3-8B \
  --served-model-name qwen3-8b \
  --devices 0,1,2,3 \
  --pp 2 \
  --tp 2 \
  --max-model-len 4096 \
  --max-num-seqs 16
```

### B1 启动命令

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

### C1 启动命令

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

## 6. 验收标准

### 6.1 正确性

- 真权重加载成功；
- `/v1/models` 返回 200；
- 首请求返回 200 和非空文本；
- greedy token IDs 与 A0/A1 基线一致；
- 若硬件数值差异导致 token 边界变化，比较 top-k logprob，阈值必须预先固定；
- 不能以 dummy 权重或“服务已启动”代替真权重首请求。

### 6.2 稳定性

- 30 分钟持续压测无 worker 退出；
- 无 HCCL timeout、OOM、fatal ACLGraph 错误；
- CPU/NPU 内存无持续单调增长；
- 重启三次均能完成真权重首请求。

### 6.3 性能

记录 P50/P95：

- TTFT；
- TPOT；
- 端到端时延；
- input/output tokens/s；
- 每 rank NPU 内存；
- 主机内存；
- H2D 时间占比；
- PP stage bubble。

PP+卸载的主要目标是“在可接受性能损失下显著降低 NPU 权重驻留”。性能损失阈值应由业务确定；建议首轮以 output throughput 下降不超过 20% 作为讨论起点，而不是默认签收值。

## 7. 当前结果

### 7.1 本地可执行验证

命令：

```bash
python -m pytest -q \
  --confcutdir=tests/ut/examples \
  tests/ut/examples/test_parallel_inference.py
```

结果：

```text
14 passed
```

覆盖：

- Dense/MoE `config.json` 判定；
- PP×TP world size 校验；
- Dense 模型误开 EP 的拒绝；
- PP×EP 命令生成；
- PP+prefetch 参数生成；
- 非法卸载分组拒绝；
- Ascend UVA functional-call 回退的 eager 约束。
- 固定 KV cache 与 `gpu-memory-utilization` 的互斥命令生成及非法值校验。

### 7.2 目标服务器环境自检

2026-07-27 执行 `./docker.sh check`，结果如下：

| 检查项 | 结果 |
|---|---|
| `npu-smi` | `25.5.0`，已显示设备健康状态 `OK` |
| vLLM | `0.25.1` |
| vLLM-Ascend | 导入成功，包未提供 `__version__` |
| torch | `2.10.0+cpu` |
| torch-npu | `2.10.0.post2` |
| 包导入路径 | `/vllm-workspace/vllm`、`/vllm-workspace/vllm-ascend`（镜像内部路径） |
| 实验目录挂载 | `/workspace/pipeline_parallel/parallel_inference.py: ok` |
| NPU 张量算子 | `npu op ok` |

结论：容器、驱动挂载、Python 软件栈和单 NPU 基础计算均通过，可以进入
Qwen3-8B 权重下载与真权重实验。

### 7.3 NPU 真权重实验结果

| ID | 启动 | 首请求 | 精度 | 稳定性 | 显存 | 吞吐 | 日志路径 |
|---|---|---|---|---|---:|---:|---|
| A0 | 通过 | 通过 | 冒烟通过 | 16/16 短压测通过 | 36047 MiB | 251.3 output tok/s | `/workspace/logs/a0_pp1_tp1_fixed_kv_graph.log` |
| A1 | 通过 | 通过 | 冒烟通过 | 16/16 短压测通过 | 54402–54681 MiB/die | 150.2 output tok/s | `/workspace/logs/a1_pp2_tp1_eager.log` |
| A2 | 通过 | 通过 | 冒烟通过 | 30 分钟、7072/7072 请求通过 | 53179–53592 MiB/die（原 eager 短测） | 34.7 output tok/s（原 eager 短测） | `/workspace/logs/a2_pp2_tp2_fixed_kv_graph_stability.log` |
| B1 | 通过 | 通过 | 冒烟通过 | 30 分钟、1184/1184 请求通过 | 53539–53744 MiB/die（原 eager 短测） | 48.9 output tok/s（原 eager 短测） | `/workspace/logs/b1_stability_30min.txt` |
| B2 | 待实测 | 待实测 | 待实测 | 待实测 | 待填 | 待填 | 待填 |
| C1 | 通过 | 通过 | 冒烟通过 | 重跑 30 分钟、1360/1360 请求通过 | 待补固定口径截图 | 49.7 output tok/s（短测） | `/workspace/logs/c1_retry2_stability_30min.txt` |

A0 单卡 fixed-KV ACLGraph 基线证据：

- 拓扑：PP=1、TP=1，使用逻辑 NPU 0，无权重卸载；
- 每 NPU KV cache 固定为 16 GiB；
- HTTP chat completion 成功，`finish_reason` 为 `stop`；
- 返回文本：`Qwen3-8B单卡基线测试成功`；
- token 使用量：prompt 28、completion 13、total 41；
- 16 请求短压测全部成功，`ok=16 fail=0`；
- 输出 goodput 为 251.3 tokens/s，P50/P95/max 延迟均约 1.0 秒；
- 实验 NPU 0 的 HBM 总占用为 36047 MiB，worker 进程显存为 32836 MiB；
- 容器主机内存约 4.724 GiB；
- 截图中逻辑 NPU 8–11 上的其他 TP/ray 任务属于服务器其他用户，不纳入统计。

A2 首请求证据：

- 拓扑：PP=2、TP=2，4 个可见 NPU，eager 模式；
- 模型：`qwen3-8b`，Qwen3-8B BF16 真权重；
- HTTP chat completion 返回成功，`finish_reason` 为 `stop`；
- 返回文本：`昇腾PP和TP二维并行测试成功`；
- `system_fingerprint` 包含 `vllm-0.25.1-tp2-pp2`；
- token 使用量：prompt 26、completion 11、total 37；
- HCCL `world_size=4`，rank 0–3 全部完成连接；
- worker 映射为 `PP0_TP0`、`PP0_TP1`、`PP1_TP0`、`PP1_TP1`；
- 仅逻辑 NPU 0–3 运行 `VLLMWorker_PP`，逻辑 NPU 4–15 无推理进程；
- 每 rank 模型权重约 3.8436 GB，KV cache 约 44.4 GiB；
- `npu-smi` 记录四个 die 的 HBM 占用为 53592、53179、53448、53186 MiB；
- 短压测参数：16 prompts、并发 4、输入约 128 tokens、固定输出 64 tokens；
- 短压测结果：16 成功、0 失败、总输出 1024 tokens、wall time 29.5 秒；
- 输出 goodput 34.7 tokens/s，单请求延迟 min 7.0 秒、P50 7.3 秒、
  P95/max 7.8 秒；
- 后续已完成 fixed-KV ACLGraph 30 分钟稳定性补测，结果见下节。

A2 fixed-KV ACLGraph 稳定性补测：

- 拓扑：PP=2、TP=2，使用逻辑 NPU 0–3，无权重卸载；
- 每 NPU KV cache 固定为 16 GiB，最大并发为 8；
- ACLGraph 服务健康检查返回 HTTP 200；
- 连续运行至少 30 分钟，共完成 442 轮；
- 每轮 16 个请求，累计 7072/7072 请求成功，0 失败；
- 每个请求固定约 128 输入 token、64 输出 token，并发为 4；
- 压测日志未检出非零 `fail`、`first fail`、`Traceback` 或 `ERROR`；
- 30 分钟控制进程按 deadline 正常结束，服务在测试结束后仍可访问。

B1 启动前证据：

- nightly CLI 已确认提供 `--offload-backend`、`--offload-group-size`、
  `--offload-num-in-group`、`--offload-prefetch-step` 和 `--offload-params`；
- dry-run 生成 PP=2、TP=1、prefetch、每四层卸载一层、提前一步预取的命令；
- `gate_up_proj` 和 `down_proj` 均作为独立参数传递给 vLLM；
- 真权重启动成功，HTTP chat completion 返回成功，`finish_reason` 为 `stop`；
- 返回文本：`PP权重卸载测试成功`；
- `system_fingerprint` 包含 `vllm-0.25.1-pp2`；
- token 使用量：prompt 22、completion 7、total 29；
- 两个 worker 均设置为 `AscendPrefetchOffloader`；
- 每 rank 加载权重约 6.5326 GB，offloader 初始化 4 个模块；
- 初始化日志报告 GPU 权重节省 1.2080 GB，静态 buffer pool 为 0.3020 GB；
- 仅物理 NPU 0 的两个 die 运行 `VLLMWorker_PP`，进程显存为
  50681、50721 MiB，HBM 总占用为 53744、53539 MiB；
- 容器 `qwen3_parallel_nightly` 的主机内存占用约 25 GiB；
- 短压测参数：16 prompts、并发 4、输入约 128 tokens、固定输出 64 tokens；
- 短压测结果：16 成功、0 失败、总输出 1024 tokens、wall time 20.9 秒；
- 输出 goodput 48.9 tokens/s，单请求延迟 min 5.0 秒、P50 5.3 秒、
  P95/max 5.6 秒；
- B1 与 A2 的 TP 数不同，不能直接用二者 HBM 差值量化卸载收益。最终显存
  对照需补跑相同 PP=2、TP=1 且关闭 offload 的 A1。

A1 首请求证据：

- 拓扑：PP=2、TP=1，2 个可见 NPU，eager 模式，无权重卸载；
- HTTP chat completion 返回成功，`finish_reason` 为 `stop`；
- 返回文本：`PP无卸载基线测试成功`；
- `system_fingerprint` 包含 `vllm-0.25.1-pp2`；
- token 使用量：prompt 24、completion 9、total 33；
- 每 rank 模型权重约 7.6576 GB，KV cache 约 41.0 GiB；
- 两个实验 die 的进程显存为 50401、50421 MiB，HBM 总占用为
  54681、54402 MiB；
- 容器 `qwen3_parallel_nightly` 的主机内存占用约 21.25 GiB；
- 短压测参数与 B1 相同：16 prompts、并发 4、输入约 128 tokens、
  固定输出 64 tokens；
- 短压测结果：16 成功、0 失败、总输出 1024 tokens、wall time 6.8 秒；
- 输出 goodput 150.2 tokens/s，单请求延迟 min 1.6 秒、P50 1.7 秒、
  P95/max 1.8 秒；
- 截图中的逻辑 NPU 8–11（`npu-smi` 表内 NPU 4、5）的
  `VLLMWorker_TP` 属于服务器其他任务，不纳入本实验统计。

### 7.4 A1 与 B1 eager 对照

| 指标 | A1：PP2×TP1，无卸载 | B1：PP2×TP1，prefetch | 变化 |
|---|---:|---:|---:|
| 每 rank 加载权重 | 7.6576 GB | 6.5326 GB | -1.1250 GB |
| 平均 HBM 总占用 | 54541.5 MiB | 53641.5 MiB | -900 MiB |
| 容器主机内存 | 21.25 GiB | 25 GiB | +3.75 GiB |
| 输出 goodput | 150.2 tok/s | 48.9 tok/s | -67.4% |
| P50 延迟 | 1.7 秒 | 5.3 秒 | +211.8% |
| P95 延迟 | 1.8 秒 | 5.6 秒 | +211.1% |

prefetch 日志报告毛节省 1.2080 GB，并分配 0.3020 GB 静态 buffer pool，
理论净节省约 0.9060 GB，与 `npu-smi` 实测平均下降约 900 MiB 一致。这证明
权重确实从 NPU 卸载到了主机侧，而不是仅接受了配置参数。

当前 B1 eager 性能损失远高于报告第 6.3 节提出的 20% 讨论起点，因此只能
签收功能与显存收益，不能签收性能。下一步应在同参数下移除
`--enforce-eager` 验证 ACLGraph+prefetch，再根据结果调整卸载比例。

### 7.5 B1 ACLGraph+prefetch 启动

移除 `--enforce-eager` 后，B1 已完成真权重加载和服务启动：

- `enable_npugraph_ex=True`，两个 PP worker 均启用 `AscendPrefetchOffloader`；
- 每 rank 加载权重约 6.5326 GB；
- offloader 仍报告节省 1.2080 GB，静态 buffer pool 为 0.3020 GB；
- torch compile 总计约 6.99 秒，模型 warmup 约 3.28 秒；
- NPU graph capture 约 7 秒，占用约 0.46 GiB graph memory；
- KV cache 约 42.2 GiB，共 614400 tokens；
- engine 初始化约 24.94 秒，其中 compilation 约 7.25 秒；
- 编译图已保存到 `/workspace/.cache/vllm/torch_compile_cache/`；
- `Application startup complete` 已出现；
- 首请求成功，返回文本为 `PP权重卸载图模式测试成功`，
  `system_fingerprint` 包含 `vllm-0.25.1-pp2`；
- token 使用量：prompt 24、completion 9、total 33；
- 两个实验 die 的进程显存为 51114、51134 MiB，HBM 总占用为
  55388、55169 MiB；
- 容器主机内存约 25.64 GiB；
- 短压测 16 成功、0 失败、总输出 1024 tokens、wall time 24.7 秒；
- 输出 goodput 41.4 tokens/s，单请求延迟 min 6.0 秒、P50 6.1 秒、
  P95/max 6.4 秒。

与 B1 eager 相比，B1 ACLGraph 的短压测吞吐下降约 15.3%，平均 HBM
增加约 1637 MiB/die，主机内存增加约 0.64 GiB。短输入、固定输出 64
tokens 的当前负载没有从图模式获益。最终图模式性能结论必须补跑相同
PP=2、TP=1、无卸载的 A1 ACLGraph，避免把图模式本身的差异误算为卸载开销。

A1 ACLGraph 无卸载基线结果：

- 首请求成功，返回文本为 `PP图模式无卸载基线成功`；
- 每 rank 模型权重约 7.6576 GB；
- KV cache 约 41.13 GiB，共 598528 tokens；
- NPU graph memory 约 0.44 GiB，engine 初始化约 14.30 秒，其中
  compilation 约 6.79 秒；
- 两个实验 die 的进程显存为 50851、50891 MiB，HBM 总占用为
  53914、53709 MiB；
- 容器主机内存约 21.73 GiB；
- 短压测 16 成功、0 失败、总输出 1024 tokens、wall time 4.7 秒；
- 输出 goodput 218.4 tokens/s，单请求延迟 min/P50/P95/max 均约
  1.1–1.2 秒。

### 7.6 A1 与 B1 ACLGraph 对照

| 指标 | A1 graph：无卸载 | B1 graph：prefetch | 变化 |
|---|---:|---:|---:|
| 每 rank 加载权重 | 7.6576 GB | 6.5326 GB | -1.1250 GB |
| KV cache | 41.13 GiB | 42.19 GiB | +1.06 GiB |
| 平均 HBM 总占用 | 53811.5 MiB | 55278.5 MiB | +1467 MiB |
| 容器主机内存 | 21.73 GiB | 25.64 GiB | +3.91 GiB |
| 输出 goodput | 218.4 tok/s | 41.4 tok/s | -81.0% |
| P50 延迟 | 1.2 秒 | 6.1 秒 | +408.3% |
| P95 延迟 | 1.2 秒 | 6.4 秒 | +433.3% |

当前使用 `gpu-memory-utilization=0.8`，vLLM 会把卸载释放的部分显存重新分配给
KV cache。B1 graph 的 KV cache 比 A1 graph 大约 1.06 GiB，同时还需要
0.3020 GB prefetch 静态 buffer 和略高的 graph/运行时开销，因此
`npu-smi` 总 HBM 不能直接反映权重卸载量，甚至出现总占用上升。

严格的 graph 显存收益测试应固定相同的 KV cache memory，再比较 A1/B1。
不过加载日志已经独立证明权重驻留从 7.6576 GB 降至 6.5326 GB，且 offloader
报告毛节省 1.2080 GB，因此功能签收不依赖总 HBM 指标。

启动器已增加 `--kv-cache-memory-bytes`。设置该参数时不再生成
`--gpu-memory-utilization`。目标服务器已完成固定 17179869184 bytes
（16 GiB/NPU）的 A1 dry-run，生成命令仅包含固定 KV 参数，互斥行为正确。
用户更正反馈：A1 ACLGraph 固定 KV 真权重启动失败，尚未出现可签收的
`Application startup complete`。根因发生在模型加载前的
`torch.npu.set_device(1)`：CANN 返回 507033 / E39007
`Starting a subprocess on the device timed out`，并提示 HDC link faulty。
这不是 KV 参数校验、模型加载或 OOM 失败；需清理本实验容器的残留设备上下文、
验证两个 die 后重试，当前不能记录固定 KV 显存结果。

随后重启实验容器 `qwen3_parallel_nightly`，分别在
`ASCEND_RT_VISIBLE_DEVICES=0,1` 下对逻辑 device 0 和 1 执行
`torch.ones(..., device="npu")` 与同步操作，两项均返回 `1.0`。这确认
HDC/设备上下文已恢复，可以在原硬件上重试，且无需将故障归因于固定 KV 参数。

使用相同 NPU 0、1 重试 A1 ACLGraph 固定 16 GiB KV 后，服务已出现
`Application startup complete`。无卸载固定 KV 基线结果：

- 每 rank 加载权重约 7.6576 GB；
- 每 NPU KV cache 固定为 16.00 GiB，共 232960 tokens；
- 两个实验 die 的进程显存为 24924、24964 MiB；
- HBM 总占用为 27992、27786 MiB，平均 27889 MiB；
- 容器 `qwen3_parallel_nightly` 的主机内存约 5.369 GiB；
- B1 固定 16 GiB KV 的 prefetch 对照已完成 dry-run 和真权重启动，
  `Application startup complete` 已出现。

当前 group size 4、每组卸载 1 层的配置在本短负载上性能损失为 81.0%，远超
20% 讨论阈值，不能作为生产配置。后续应依次验证固定 KV cache、降低卸载比例、
增加 prefetch step，以及更长输出和更高并发负载。

### 7.7 固定 16 GiB KV 的显存对照

A1 与 B1 均使用 ACLGraph、PP=2、TP=1、NPU 0/1、最大长度 4096、
最大并发 8，并固定每 NPU KV cache 为 17179869184 bytes。

| 指标 | A1 fixed-KV：无卸载 | B1 fixed-KV：prefetch | 变化 |
|---|---:|---:|---:|
| 每 rank 加载权重 | 7.6576 GB | 6.5326 GB | -1.1250 GB |
| KV cache | 16.00 GiB | 16.00 GiB | 0 |
| KV cache token 容量 | 232960 | 232960 | 0 |
| die 0 HBM | 27992 MiB | 27135 MiB | -857 MiB |
| die 1 HBM | 27786 MiB | 26928 MiB | -858 MiB |
| 平均 HBM | 27889 MiB | 27031.5 MiB | -857.5 MiB（-3.1%） |
| 平均 worker 进程显存 | 24944 MiB | 24085 MiB | -859 MiB |
| 容器主机内存 | 5.369 GiB | 9.259 GiB | +3.890 GiB |

B1 日志再次确认两个 worker 使用 `AscendPrefetchOffloader`，初始化 4 个
卸载模块，报告 GPU 权重毛节省 1.2080 GB、静态 buffer pool 0.3020 GB。
在固定 KV 后，worker 进程显存下降约 859 MiB，与毛节省扣除静态 buffer
和运行时开销后的预期一致。

这组结果完成了 PP+权重卸载的显存功能签收：权重驻留、worker 进程显存和
NPU HBM 均下降，主机内存相应增加。性能签收仍未通过，因为当前卸载比例在
短请求负载上的吞吐损失显著高于 20% 讨论阈值。

### 7.8 B1 prefetch 30 分钟稳定性

B1 使用 PP=2、TP=1、ACLGraph、prefetch 权重卸载和固定 16 GiB KV/NPU
持续压测至少 30 分钟：

- 测试从容器时间 07:49:14 开始，最后一轮从 08:19:01 开始并正常完成；
- 共完成 74 轮，每轮 16 个请求，累计 1184/1184 请求成功；
- 每个请求固定约 128 输入 token、64 输出 token，并发为 4；
- 各轮均为 `ok=16 fail=0`，输出 goodput 稳定在约 40–42 tokens/s；
- 压测日志未检出非零 `fail`、`first fail`、`free()`、`Traceback` 或 `ERROR`；
- 压测控制进程按 deadline 自然结束，测试后 `/health` 仍返回 HTTP 200；
- 宿主机 Python/`tee` 曾出现 glibc `free(): invalid next size (fast)`，因此最终
  压测客户端通过 `docker exec -d` 放入实验容器运行。该异常不属于 vLLM
  服务端或 NPU worker，容器内完整稳定性测试未复现。

### 7.9 C1 Qwen3-30B-A3B PP×EP

C1 使用完整的 57 GiB Qwen3-30B-A3B MoE 权重，模型配置包含 128 个专家、
每 token 路由 8 个专家。首次在逻辑 NPU 0–3 上启动时，其中一个目标设备仅余
14.73/61.28 GiB，未达到默认 `gpu-memory-utilization=0.8` 要求的
49.02 GiB，因此在模型加载前被容量检查拒绝；这不是模型或通信失败。

随后改用空闲逻辑 NPU 2–5，并固定每 NPU 8 GiB KV cache、最大并发 4：

- 拓扑为 PP=2、TP=2、world size=4，并启用 expert parallel；
- 四个 rank 均完成 HCCL 初始化，日志显示 PP、TP 和 EP rank 分配；
- 真权重加载和 eager 服务启动成功；
- 首请求 HTTP 200，返回 `Qwen3-30B-A3B的PP和EP测试成功`；
- `system_fingerprint` 包含 `tp2-pp2-ep`；
- 16/16 短压测成功，输出 goodput 为 49.7 tokens/s；
- 长稳测试前 75 轮均为 16/16 成功，即至少 1200 个请求成功；
- 之后服务在仍有 4 个运行请求时进入完整的 FastAPI/MPClient 优雅关闭流程，
  压测末轮为 12 成功、4 超时；
- Docker 容器始终为 running，`restart_count=0`，对应时段无容器重启事件；
- 服务日志未出现 OOM、Python traceback、WorkerProc failure 或 HCCL fatal。

第一次长稳结果记为“外部终止信号中断”，不归类为模型或并行框架失败。
随后使用相同 PP=2、TP=2、EP、8 GiB KV/NPU 和 eager 参数完整重跑：

- 连续运行至少 30 分钟并自然结束；
- 共完成 85 轮，每轮 16 个请求，累计 1360/1360 成功；
- 稳定性日志未检出非零 `fail`、`first fail`、`free()`、`Traceback`、
  `ERROR` 或 HCCL fatal；
- 测试结束后压测进程自然消失，服务 `/health` 仍返回 HTTP 200。

因此 C1 的 PP×EP 功能、短压测和连续 30 分钟稳定性均正式签收为通过。

## 8. 风险与排查顺序

1. 模型变体不明确：先记录 `architectures`、dtype、量化配置和权重索引。
2. 设备拓扑错误：先验证可见设备数和 rank 映射，再查 HCCL。
3. false-ready：服务启动后必须发首请求。
4. 图模式问题：同参数切换 eager；eager 通过后再定位 ACLGraph。
5. PP 气泡：小 batch/低并发可能让 PP 比单 rank 更慢，这是调度问题，不是正确性失败。
6. H2D 成为瓶颈：减少卸载层数或只卸载大 MLP/专家矩阵。
7. 主机内存/NUMA：pinned CPU 权重和静态 buffer 需要容量与亲和性规划。
8. EP 负载不均：扩大 MoE 模型时记录专家 token 分布，再决定是否启用 EPLB。

## 9. 放大模型的决策门

只有以下条件全部满足才进入更大模型：

- A2 正确性与 30 分钟稳定性通过；
- B1 明确测得每 rank NPU 内存下降；
- B1/B2 性能损失在业务阈值内；
- C1 的所有 rank 专家通信正常；
- 已保存可复现命令、版本、日志和结果 JSON。

当前决策分为两类：

- **功能验证 GO**：Qwen3-8B 的 PP×TP、PP+prefetch，以及放大到
  Qwen3-30B-A3B 后的 PP×EP 均已通过，A2、B1、C1 均完成 30 分钟稳定性；
- **prefetch 生产性能 NO-GO**：当前 group size 4 的卸载配置在短请求负载下
  性能损失明显超过 20% 讨论阈值。上线前仍需降低卸载比例、调整 prefetch
  step，并按实际业务输入/输出长度重新压测。

推荐放大顺序：

```text
Qwen3-8B Dense
  → Qwen3-30B-A3B（MoE，验证 EP）
  → Qwen3-32B Dense（验证纯容量）
  → Qwen3-235B-A22B 或目标大模型
```

Dense 与 MoE 两条线要分开归因：Dense 放大主要验证 PP/TP 和容量，MoE 放大还增加专家通信与负载均衡风险。

## 10. 本轮签收记录

- PP×TP：Qwen3-8B、PP=2、TP=2、四个 rank、真权重首请求通过；
- PP×TP eager 短压测：16/16 成功，34.7 output tok/s；
- PP 无卸载 eager 基线：PP=2、TP=1，150.2 output tok/s；
- PP+prefetch eager：首请求通过，48.9 output tok/s，功能通过但性能未达标；
- PP 无卸载 ACLGraph 基线：218.4 output tok/s；
- PP+prefetch ACLGraph：首请求通过，41.4 output tok/s，功能通过但性能未达标；
- fixed-KV 显存签收：16 GiB KV/NPU，prefetch 平均降低
  857.5 MiB/die HBM、859 MiB/worker 进程显存，主机内存增加 3.890 GiB；
- 关键日志位于宿主机
  `/home/vllm/l00977701/runtime/logs/` 与
  `/home/vllm/l00977701/runtime/evidence/`；
- A2 30 分钟稳定性：442 轮、7072/7072 请求成功；
- B1 30 分钟稳定性：74 轮、1184/1184 请求成功；
- PP×EP：Qwen3-30B-A3B、PP=2、TP=2、EP on，真权重首请求及
  16/16 短压测通过，短测吞吐 49.7 output tok/s；
- C1 30 分钟稳定性重跑：85 轮、1360/1360 请求成功，测试后服务健康；
- 全部实验使用 Qwen3-8B 或 Qwen3-30B-A3B 真权重，不使用 dummy 权重；
- EP 对 Dense Qwen3-8B 不适用，已在 Qwen3-30B-A3B MoE 上完成验证；
- 当前本地变更未提交，因此无最终 commit hash。
