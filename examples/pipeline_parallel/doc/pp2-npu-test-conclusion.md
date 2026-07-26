# PP 并行运行指南(vllm-ascend / A3)

> 环境:A3 节点(8×Ascend910,每卡 2 die = davinci0..15),镜像 `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3-openeuler`(实际 vllm-ascend `0.23.0rc2.dev6`),容器 `vllm_dynamic`(`--net=host`)。

## 1. 怎么跑 PP 并行

### 用 docker.sh(推荐,自动选卡)

```bash
# tp=1 pp=2(PP 走 SERVE_EXTRA,选卡用 SERVE_TP=2 选 2 个 die)
SERVE_MODEL=<模型路径> \
SERVE_TP=2 \
SERVE_EXTRA="--pipeline-parallel-size 2 --max-model-len 4096" \
./docker.sh serve
```
- `SERVE_TP=2` = 自动选 2 个空闲 die(`find_idle_npu.sh --pick 2`,strict 只选无进程的 die)。
- `--same-card` 默认在 `SERVE_TP>=2` 时自动开(`FORCE_SAME_CARD=auto`),优先同卡双 die。
- 非 eager(默认):带 cudagraph capture;加 `--enforce-eager` 走 eager(无 capture,启动快)。

### 手动起(最直接)

```bash
docker exec -e ASCEND_RT_VISIBLE_DEVICES=2,3 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  vllm_dynamic bash -lc "vllm serve <模型路径> \
  --tensor-parallel-size 1 --pipeline-parallel-size 2 \
  --host 0.0.0.0 --port 8002 --max-model-len 4096"
```
- `ASCEND_RT_VISIBLE_DEVICES` 两个 die:同卡(如 `2,3`)或跨卡(如 `2,6`)均可。
- 去掉/加上 `--enforce-eager` 切换 eager / 非 eager。
- 模型路径用本地快照目录(如 `/root/.cache/modelscope/models/Qwen--Qwen3-30B-A3B/snapshots/master`),`HF_HUB_OFFLINE=1` 必带(huggingface 被墙)。

### 选卡

```bash
./find_idle_npu.sh --pick 2            # 任选 2 个空闲 die
./find_idle_npu.sh --pick 2 --same-card # 强制同卡双 die(走 HCCS,低延迟)
./find_idle_npu.sh --list               # 列全部空闲 die
```

## 2. 注意事项

1. **不要 apply `0001-fix-pp-...patch`**:两处改动不必要,改动 2 还会把默认 `FULL_AND_PIECEWISE` 降级成纯 FULL、砍掉已工作的 PIECEWISE 优化。
2. **`HCCL_SOCKET_IFNAME` 不用设**:同卡/跨卡、带/不带都验过能跑。
3. **同卡 vs 跨卡**:都能跑;同卡走 HCCS 片内链路延迟低,跨卡走 host NIC 带宽有限。生产建议优先同卡或上 RoCE。
4. **非 eager 默认 `cudagraph_mode=FULL_AND_PIECEWISE`**:PIECEWISE(prefill-decode)+ FULL(decode)两套 capture,正常会都过。若遇 `507903 rtStreamEndCapture` 才需要排查(目前未复现)。
5. **模型路径**:modelscope 模型在宿主 `/root/.cache/modelscope`,`docker.sh` 已默认挂载进容器;手动起别的容器要自己挂。
6. **崩了怎么查**:若复现 `EI0015`(rank 连不上 root),第一时间记录:
   - `ASCEND_RT_VISIBLE_DEVICES`(用了哪两个 die);
   - `npu-smi info -t proc-mem -i <卡> -c <die>`(残留进程占着?);
   - 机器负载 / `HCCL_CONNECT_TIMEOUT`。
   别在没这三样时瞎猜根因。
