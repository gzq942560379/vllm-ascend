#!/usr/bin/env bash
# Launch vLLM with pipeline-parallel-size=2 on Ascend NPU.
# 前提:vllm + vllm-ascend 已装、CANN/ATB env 已 source(容器内 entrypoint 通常已做)。
#
# 用法:
#   ./launch_pp_serve.sh <model_path> <die1>,<die2> [port] [extra vllm args...]
# 例:
#   ./launch_pp_serve.sh /root/.cache/modelscope/models/Qwen--Qwen3-30B-A3B/snapshots/master 2,3 8000
#   ./launch_pp_serve.sh <model> 2,3 8000 --enforce-eager            # eager(无 cudagraph capture,启动快)
#   ./launch_pp_serve.sh <model> 2,3 8000 --max-model-len 8192
#
# die 选择:同卡双 die(如 2,3 = 卡1 的两个 die)走 HCCS,延迟最低;跨卡(如 2,6)也能跑。
# 昇腾 die 编号:davinci0..15,每卡 2 die(卡0=0,1;卡1=2,3;...)。用 `npu-smi info -m` 看映射。
set -euo pipefail

MODEL="${1:?usage: $0 <model_path> <die1,die2> [port] [extra vllm args...]}"
DIES="${2:?die list, e.g. 2,3 (同卡) 或 2,6 (跨卡)}"
PORT="${3:-8000}"
shift 3 || true
EXTRA="${*:-}"

export ASCEND_RT_VISIBLE_DEVICES="$DIES"
export HF_HUB_OFFLINE=1            # huggingface 被墙时必带;模型用本地快照路径
export TRANSFORMERS_OFFLINE=1

echo "[launch_pp] model=$MODEL dies=$DIES port=$PORT extra=[$EXTRA]"
exec vllm serve "$MODEL" \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 2 \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len 4096 \
  $EXTRA
