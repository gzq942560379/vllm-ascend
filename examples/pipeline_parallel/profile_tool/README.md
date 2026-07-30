# vLLM-Ascend 二维并行性能边界测试工具

该目录包含两套互不冲突的工具：

- `profile_trace.ps1`：保留原有单 NPU、单次或多请求 Profiling 流程。
- `parallel_bench.ps1`：新的 PP×TP/EP 性能边界测试平台，预留 CP 接口。

新平台从 Windows 提交任务，但控制器在昇腾服务器后台运行。SSH 断开、
Windows 休眠或 NPU 暂时被占用不会让任务丢失。控制器不会停止、抢占或杀死
其他用户的 NPU 进程；资源不足时默认每 60 秒重试，最长等待 6 小时。

## 1. 默认配置

| 项目 | 默认值 |
|---|---|
| 服务器 | `192.168.13.190` |
| 容器 | `qwen3_parallel_nightly` |
| 模型 | `/models/Qwen3-30B-A3B` |
| 模型类型 | Qwen3 MoE |
| 执行模式 | ACLGraph |
| NPU | 8 张 910B、最多 16 个 die |
| 资源等待 | 60 秒轮询，最长 21600 秒 |
| Windows 输出 | `D:\vllm-parallel-bench\<run_id>` |

所有常用参数统一放在：

`configs/parallel_bench_config.json`

包括服务器、SSH 端口、容器、版本、模型、矩阵、工作负载、Profiling、
远端目录和 Windows 输出目录。工具不会保存 SSH 密码。每次操作会建立一条
OpenSSH 复用连接，因此使用密码时只输入一次；配置 OpenSSH 密钥或
`ssh-agent` 后可以完全免输密码。

## 2. 快速矩阵

`-Matrix quick` 会依次执行：

| Case | PP | TP | EP | die 数 | 用途 |
|---|---:|---:|:---:|---:|---|
| R0 | 1 | 1 | off | 1 | 单 die 容量参考，装不下记为 capacity skip |
| P1 | 2 | 1 | off | 2 | PP 基线 |
| T1 | 2 | 2 | off | 4 | PP×TP |
| E1 | 2 | 2 | on | 4 | PP×EP 对照 |
| T2 | 4 | 2 | off | 8 | 更深 PP |
| E2 | 4 | 2 | on | 8 | 更深 PP+EP |

`-Matrix boundary` 扩展 `PP ∈ {2,4,8}`、`TP ∈ {1,2,4}`，并对 MoE
模型比较 EP on/off；只保留 world size 不超过 16 的点。

全矩阵运行轻量性能测试。Profiling 只运行代表点和边界点，避免 Profiler
扰动所有性能结果。

## 3. 第一次提交

在 Windows PowerShell 执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
cd E:\vllm\vllm-ascend\examples\pipeline_parallel\profile_tool

.\parallel_bench.ps1 -Action Submit `
  -ConfigFile .\configs\parallel_bench_config.json
```

输出会显示 `RunId`。提交完成后可以关闭 SSH 或 Windows 终端。

默认工作负载为：

- 输入 token：128、512、2048
- 输出 token：64、256
- 并发：1、4、16、32
- 每点 64 请求、3 次重复、4 请求预热
- Profiling：16 个真实请求，并发 4，输入 512、输出 64

先做小规模烟测：

```powershell
.\parallel_bench.ps1 -Action Submit `
  -ConfigFile .\configs\qwen3_30b_a3b_smoke.json
```

该烟测只跑 `PP=2, TP=1`、2 个短请求，不启用 Profiling。确认完整生命周期
通过后，再提交默认 quick 矩阵。

## 4. 查询、断点续跑和下载

```powershell
# 查询状态
.\parallel_bench.ps1 -Action Status `
  -ConfigFile .\configs\parallel_bench_config.json

# 控制器异常退出后，从 state.json 的未完成点继续
.\parallel_bench.ps1 -Action Resume `
  -ConfigFile .\configs\parallel_bench_config.json

# 下载到 client.output_dir
.\parallel_bench.ps1 -Action Fetch `
  -ConfigFile .\configs\parallel_bench_config.json

# 安全请求取消；控制器会在检查点停止它自己启动的服务
.\parallel_bench.ps1 -Action Cancel `
  -ConfigFile .\configs\parallel_bench_config.json
```

当 `run.run_id` 为 `auto` 时，提交成功后会在配置文件旁写入
`<配置文件>.last-run.json`。后续 Status、Resume、Fetch 和 Cancel 自动读取
最近一次 RunId，不必再复制到命令行。若要操作指定历史任务，可把
`run.run_id` 改为对应值，或临时传入 `-RunId`。

`state.json` 中常见状态：

- `WAIT_NPU`：等待足够空闲 die，属于正常排队。
- `PREPARE/WARMUP/BENCHMARK/PROFILE/ANALYZE`：正在执行。
- `RETRYABLE`：单点遇到瞬态故障，将保留结果后重试。
- `SKIPPED_CAPACITY`：模型在该并行配置下装不下，不算工具故障。
- `SKIPPED_UNSUPPORTED`：当前 vLLM 版本没有对应 CLI 能力。
- `COMPLETE/FAILED/CANCELLED`：终态。

## 5. 切换模型、版本和模式

直接编辑 `configs/parallel_bench_config.json` 中的 `container`、`model` 和
`execution`。例如将 `container.name` 改成 `qwen3_v0251`，
`model.path` 改成 `/models/Qwen3-8B`，`model.kind` 改成 `dense`，
`execution.execution_mode` 改成 `eager`。保存后仍只执行同一条 Submit 命令。

Dense 模型会跳过 EP 点。工具启动前记录容器内 vLLM、vLLM-Ascend、PyTorch
和 torch-npu 版本，并通过 `vllm serve --help=all` 探测参数，而不是只根据
版本号猜测功能。

执行模式区别：

- `aclgraph`：接近生产推理性能，适合吞吐、TTFT 和 TPOT 对比。
- `eager`：算子边界更清楚，适合逐算子诊断，但不能与 ACLGraph 吞吐直接等价。

## 6. 自定义 PP/TP/EP/CP

默认可编辑配置文件在：

`configs/parallel_bench_config.json`

需要精确矩阵时，在 spec 的 `matrix.cases` 中写：

```json
{
  "matrix": {
    "preset": "custom",
    "cases": [
      {"case_id": "custom-p2t4", "pp": 2, "tp": 4, "ep": false, "cp": 1, "profile": true},
      {"case_id": "custom-p4t2e", "pp": 4, "tp": 2, "ep": true, "cp": 1, "profile": true}
    ]
  }
}
```

保存为本地 JSON 后提交：

```powershell
.\parallel_bench.ps1 -Action Submit -ConfigFile .\my_parallel_spec.json
```

CP 已进入 schema 和命令能力适配器。当容器的 `vllm serve --help=all` 没有
CP 参数时，对应点会标记 `SKIPPED_UNSUPPORTED`，不会拖垮整个矩阵。

## 7. 指标口径

- **TTFT**：HTTP 请求发出到第一个非空 SSE 输出块到达的时间。
- **TPOT**：`(请求完成时间 - 首 token 时间) / (输出 token 数 - 1)`。
- **ITL**：相邻流式输出到达间隔，报告 p50/p95/p99。
- **吞吐**：所有成功请求输出 token 总数除以该并发批次真实 wall time。
- **PP 空泡率**：`1 - stage active compute / measurement makespan`。
- **计算/访存/通信占比**：根据 `kernel_details.csv` 的执行时间分类。
- **通信延迟**：区分 Send/Recv、AllReduce、AllGather、ReduceScatter、
  AllToAll/dispatch/combine，并统计次数、总时长、p50、p95、最大值。

计算/访存占比首先是 kernel 时间分类口径。只有 CANN 输出确实包含 HBM/LLC/PMU
字段时，报告才会额外给出硬件利用率；缺少字段时明确标记 unavailable。

## 8. 输出结果

```text
D:\vllm-parallel-bench\<run_id>\
├── spec.json
├── environment.json
├── state.json
├── summary.csv
├── report.md
├── report.html
├── charts\
│   ├── throughput_scaling.svg
│   ├── ttft_tpot_comparison.svg
│   ├── bubble_by_stage.svg
│   ├── compute_memory_comm_share.svg
│   ├── communication_breakdown.svg
│   └── scaling_efficiency.svg
├── cases\<case_id>\request_metrics.jsonl
└── profiles\<case_id>\...\ASCEND_PROFILER_OUTPUT\
    ├── trace_view.json
    ├── kernel_details.csv
    └── operator_details.csv
```

Windows 端可直接用 MindStudio Insight 打开各 rank 的 `trace_view.json`。
`report.html` 和 SVG 不依赖 Python 绘图库，可直接用浏览器打开。

## 9. 安全和故障边界

- 在 Docker 宿主机运行 Windows 命令，不要进入容器后执行 `docker`。
- 工具使用每个 die 的 `flock` 防止本工具自身竞争，同时用 `npu-smi` 二次确认。
- 锁无法约束不使用该工具的其他用户，因此启动前和等待期间仍会重新检查占用。
- 每个 run 使用独立目录、端口、PID 文件和日志。
- 停止服务时只读取当前 run 自己写入的 PID，不使用 `pkill vllm`。
- SSH 密码、私钥和令牌不会进入 spec、日志或报告。

如果控制器意外退出，先看：

```powershell
.\parallel_bench.ps1 -Action Status -RunId <run_id>
ssh root@192.168.13.190 "tail -100 /home/vllm/l00977701/runtime/parallel_bench_runs/<run_id>/controller.log"
```

然后使用 `Resume`。已经 `COMPLETE` 或已跳过的 case 不会重复运行。
