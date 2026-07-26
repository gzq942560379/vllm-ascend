# Pipeline Parallel (PP=2) on Ascend NPU — Example

在昇腾(A3,8×Ascend910,每卡 2 die)上用 vLLM 跑 pipeline-parallel-size=2 的最小可运行示例。已实测 Qwen3-30B-A3B 通过。更多说明见 `doc/pp2-npu-test-conclusion.md`。

## 文件

| 文件 | 作用 |
|---|---|
| `launch_pp_serve.sh` | 起 PP=2 的 vLLM serve(指定模型 + 两个 die + 端口 + 额外参数) |
| `bench_serve.py` | 轻量在线吞吐压测(纯 stdlib,打 `/v1/completions`) |
| `docker.sh` | 容器管理:起/停/登录/serve vllm-ascend 容器(`--net=host`,透传 davinci,选卡) |
| `find_idle_npu.sh` | 选空闲 die(strict 查 `npu-smi -t proc-mem` 无进程的 die),支持 `--same-card` 强制同卡 |
| `install_sources.sh` | 容器内从源码编 vllm + vllm-ascend(editable,落回宿主挂载目录,抗 recreate) |

> `docker.sh` / `find_idle_npu.sh` / `install_sources.sh` 是环境搭建脚本(非 PP 专属),PP 例子跑在它们搭好的容器之上。

## 跑起来

### 1. 起 PP=2 serve

```bash
# 同卡双 die(卡1 的 die2,3,走 HCCS 延迟最低)
./launch_pp_serve.sh /root/.cache/modelscope/models/Qwen--Qwen3-30B-A3B/snapshots/master 2,3 8000

# eager 模式(无 cudagraph capture,启动快、调试用)
./launch_pp_serve.sh <model> 2,3 8000 --enforce-eager

# 非 eager(默认):带 cudagraph capture,吞吐更高;默认 cudagraph_mode=FULL_AND_PIECEWISE
```

容器外手动起(等价):

```bash
docker exec -e ASCEND_RT_VISIBLE_DEVICES=2,3 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  <container> bash -lc "vllm serve <model> \
  --tensor-parallel-size 1 --pipeline-parallel-size 2 \
  --host 0.0.0.0 --port 8000 --max-model-len 4096"
```

### 2. 压测吞吐

```bash
python bench_serve.py --url http://127.0.0.1:8000 \
  --model <model_path> \
  --num-prompts 64 --concurrency 16 --in-len 256 --out-len 256
```

输出 `THROUGHPUT (output tok/s, goodput)`、per-req latency(p50/p95/max)。

## die 怎么选

- 昇腾 die 编号 `davinci0..15`,每卡 2 die:**卡0=0,1;卡1=2,3;卡2=4,5;...** 用 `npu-smi info -m` 看映射。
- **同卡双 die**(如 `2,3`)走 HCCS 片内链路,延迟最低,**优先选**。
- **跨卡**(如 `2,6`)也能跑,走 host NIC,带宽有限;生产跨卡建议上 RoCE。
- 选空闲 die:看 `npu-smi info -t proc-mem -i <卡> -c <die>` 有没有进程占着,挑无进程的。

## 注意事项

1. **`HCCL_SOCKET_IFNAME` 不用设**:同卡/跨卡、带/不带都验过能跑。
2. **非 eager 默认 `cudagraph_mode=FULL_AND_PIECEWISE`**:PIECEWISE(prefill-decode)+ FULL(decode)两套 capture 正常都会过;若遇 `507903 rtStreamEndCapture` 才需排查(目前未复现)。
3. **`HF_HUB_OFFLINE=1` 必带**(huggingface 被墙),模型用本地快照目录路径。
4. **崩了查什么**:若复现 `EI0015`(rank 连不上 root),记录 `ASCEND_RT_VISIBLE_DEVICES`、`npu-smi -t proc-mem` 残留进程、机器负载 / `HCCL_CONNECT_TIMEOUT`,再断根因,别瞎猜。
