#!/bin/bash
# vllm-ascend 容器：开启 / 关闭 / 登录 / 起 vllm 服务
# 默认镜像: quay.io/ascend/vllm-ascend:nightly-main-a3
# 2026-07-27 服务器本地镜像 ID: a7788a65d91b
# 镜像 entrypoint 会自动 source CANN/ATB set_env.sh。
# 默认 dynamic 模式：透传全部 /dev/davinciX，运行时再设 ASCEND_RT_VISIBLE_DEVICES 选卡。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# 可配置参数（环境变量可覆盖）
# ---------------------------------------------------------------------------
USER_TAG="${USER_TAG:-vllm}"
IMAGE="${IMAGE:-quay.io/ascend/vllm-ascend:nightly-main-a3}"
# 独立实验包布局。把整个 pipeline_parallel 目录复制到 BASE_DIR 下即可。
# 例如：/home/vllm/l00977701/pipeline_parallel
BASE_DIR="${BASE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
BUNDLE_DIR="${BUNDLE_DIR:-${SCRIPT_DIR}}"
RUNTIME_DIR="${RUNTIME_DIR:-${BASE_DIR}/runtime}"
WORK_DIR="${WORK_DIR:-/workspace}"
MODEL_DIR="${MODEL_DIR:-${BASE_DIR}/models/Qwen3-8B}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${BASE_DIR}/huggingface-cache}"
# 仅 NPU_MODE=static 时使用（示例：NPU3 双 die -> logic 6,7）
ASCEND_DEVICES="${ASCEND_DEVICES:-6,7}"
#   dynamic - 透传全部 /dev/davinciX，执行命令时再设 ASCEND_RT_VISIBLE_DEVICES
#   static  - 按 ASCEND_DEVICES 透传并固定 ASCEND_VISIBLE_DEVICES
NPU_MODE="${NPU_MODE:-dynamic}"
SHM_SIZE="${SHM_SIZE:-32g}"
RECREATE="${RECREATE:-0}"

# 容器名
if [[ "${NPU_MODE}" == "dynamic" ]]; then
  CONTAINER_NAME="${CONTAINER_NAME:-${USER_TAG}_dynamic}"
else
  _DEVICES="${ASCEND_DEVICES//[[:space:]]/}"
  _DEVICES="${_DEVICES//,/_}"
  CONTAINER_NAME="${CONTAINER_NAME:-${USER_TAG}_${_DEVICES}}"
fi

# vllm serve 默认参数（serve 子命令用）
SERVE_MODEL="${SERVE_MODEL:-/models}"
SERVE_TP="${SERVE_TP:-1}"                 # 自动选卡挑几个 die(tp)
SERVE_PICK_FLAGS="${SERVE_PICK_FLAGS:---wait}"  # find_idle_npu.sh 选卡参数;默认 --wait 无空闲时阻塞等(find_idle_npu 永远 strict)
FIND_IDLE_NPU="${FIND_IDLE_NPU:-${SCRIPT_DIR}/find_idle_npu.sh}"
SERVE_PORT="${SERVE_PORT:-8000}"
SERVE_EXTRA="${SERVE_EXTRA:-}"            # 额外 vllm 参数，例：--enforce-eager --max-model-len 4096
# 多 die(tp>=2 或 pp>=2)时优先同卡选 die(走 HCCS,低延迟);非 correctness 修复,跨卡也能跑。
FORCE_SAME_CARD="${FORCE_SAME_CARD:-auto}"   # auto: SERVE_TP>=2 时开;1/0 可强制

usage() {
  cat <<EOF
Usage: ./docker.sh [command]

Image:     ${IMAGE}
Container: ${CONTAINER_NAME}
  Bundle:    ${BUNDLE_DIR} -> /workspace/pipeline_parallel
  Runtime:   ${RUNTIME_DIR} -> /workspace
  Models:    ${MODEL_DIR} -> /models

Commands:
  (none)  启动容器（若未运行）并登录 shell
  start   仅启动容器
  restart 重建容器
  stop    停止容器
  shell   登录 shell（同默认）
  serve   起一个 vllm serve 进程（前台）
  logs    查看日志
  check   容器内跑 npu-smi + import 自检

Config: 见本文件顶部，或 export 环境变量覆盖

NPU binding (默认 dynamic):
  ./docker.sh start
  # 执行时再选卡（例：NPU3 双 die -> logic 6,7）
  docker exec -it -e ASCEND_RT_VISIBLE_DEVICES=6,7 ${CONTAINER_NAME} bash

  # 固定卡启动（static）:
  NPU_MODE=static ASCEND_DEVICES=6,7 RECREATE=1 ./docker.sh restart

serve:
  SERVE_MODEL=/models ./docker.sh serve
  # 永远用 find_idle_npu.sh 自动选空闲卡(strict,无进程的 die),无空闲则 --wait 阻塞等。不手动指定卡。
  #   SERVE_TP=2 ./docker.sh serve                       # 挑 2 个 die(tp=2)
  #   SERVE_PICK_FLAGS="--wait-seconds 30" ./docker.sh serve  # 自定义等待轮询
  #   SERVE_EXTRA="--enforce-eager --max-model-len 4096" ./docker.sh serve  # 额外 vllm 参数
EOF
}

container_exists() { docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; }
container_running() { docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; }

validate_model_dir() {
  if [[ ! -s "${MODEL_DIR}/config.json" ]]; then
    echo "Model config not found on host: ${MODEL_DIR}/config.json" >&2
    exit 1
  fi
}

create_container() {
  local device_args=() env_args=()

  mkdir -p "${RUNTIME_DIR}" "${HF_CACHE_DIR}"
  validate_model_dir

  case "${NPU_MODE}" in
    static)
      IFS=',' read -ra dev_ids <<< "${ASCEND_DEVICES}"
      for id in "${dev_ids[@]}"; do
        id="${id//[[:space:]]/}"
        [[ -n "${id}" ]] || continue
        device_args+=(--device="/dev/davinci${id}")
      done
      env_args+=(-e "ASCEND_VISIBLE_DEVICES=${ASCEND_DEVICES}")
      ;;
    dynamic)
      shopt -s nullglob
      for dev in /dev/davinci[0-9]*; do device_args+=(--device="${dev}"); done
      shopt -u nullglob
      ;;
    *) echo "Invalid NPU_MODE='${NPU_MODE}', expected: static|dynamic" >&2; exit 1 ;;
  esac

  echo "Creating container: ${CONTAINER_NAME} (image: ${IMAGE}, NPU_MODE=${NPU_MODE})"
  docker run -d \
    --name "${CONTAINER_NAME}" \
    --restart unless-stopped \
    --net=host \
    --shm-size "${SHM_SIZE}" \
    --privileged \
    -w "${WORK_DIR}" \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    --device=/dev/davinci_mini_manage \
    "${device_args[@]}" \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
    -v "${RUNTIME_DIR}:/workspace" \
    -v "${BUNDLE_DIR}:/workspace/pipeline_parallel:ro" \
    -v "${MODEL_DIR}:/models" \
    -v "${HF_CACHE_DIR}:${WORK_DIR}/.cache/huggingface" \
    -e "HOME=${WORK_DIR}" \
    "${env_args[@]}" \
    "${IMAGE}" \
    /bin/bash -c "sleep infinity"
}

ensure_running() {
  if container_running; then return 0; fi
  if container_exists; then
    if [[ "${RECREATE}" == "1" ]]; then
      echo "Removing existing container: ${CONTAINER_NAME}"; docker rm -f "${CONTAINER_NAME}"
    else
      echo "Starting container: ${CONTAINER_NAME}"; docker start "${CONTAINER_NAME}"; return 0
    fi
  fi
  create_container
}

cmd_shell() { ensure_running; exec docker exec -it -e "HOME=${WORK_DIR}" -w "${WORK_DIR}" "${CONTAINER_NAME}" bash -i; }
cmd_start() { ensure_running; echo "Container '${CONTAINER_NAME}' is running (image: ${IMAGE})."; }
cmd_restart() {
  validate_model_dir
  if container_exists; then echo "Removing existing container: ${CONTAINER_NAME}"; docker rm -f "${CONTAINER_NAME}"; fi
  create_container; echo "Container '${CONTAINER_NAME}' recreated (image: ${IMAGE})."
}
cmd_stop() {
  if ! container_exists; then echo "Container '${CONTAINER_NAME}' not found."; exit 1; fi
  if container_running; then docker stop "${CONTAINER_NAME}"; echo "Container '${CONTAINER_NAME}' stopped."
  else echo "Container '${CONTAINER_NAME}' is not running."; fi
}
cmd_logs() { if ! container_exists; then echo "Container '${CONTAINER_NAME}' not found. Run: ./docker.sh start"; exit 1; fi; exec docker logs -f "$@" "${CONTAINER_NAME}"; }

cmd_serve() {
  ensure_running
  local devices tp src
  if [[ ! -x "${FIND_IDLE_NPU}" ]]; then
    echo "find_idle_npu.sh 不可执行: ${FIND_IDLE_NPU}" >&2; exit 1
  fi
  # 多 die 时强制同卡选 die(auto 模式:SERVE_TP>=2 开)
  if [[ "${FORCE_SAME_CARD}" == "auto" ]]; then
    if [[ "${SERVE_TP}" -ge 2 ]]; then FORCE_SAME_CARD=1; else FORCE_SAME_CARD=0; fi
  fi
  export FORCE_SAME_CARD
  local same_card_flag=()
  [[ "${FORCE_SAME_CARD}" == "1" ]] && same_card_flag=(--same-card)
  src="auto(find_idle_npu.sh --pick ${SERVE_TP} ${SERVE_PICK_FLAGS}${same_card_flag:+ ${same_card_flag[*]}})"
  devices="$("${FIND_IDLE_NPU}" --pick "${SERVE_TP}" ${SERVE_PICK_FLAGS} ${same_card_flag[@]})" || {
    echo "自动选卡失败(被中断或 npu-smi 出错)。" >&2; exit 1
  }
  [[ -n "${devices}" ]] || { echo "find_idle_npu.sh 返回空(无空闲 die 且未 --wait 或等待超时)。" >&2; exit 1; }
  tp=$(echo "${devices}" | awk -F, '{print NF}')
  echo "Serving ${SERVE_MODEL} on NPU logic ${devices} (tp=${tp}, same_card=${FORCE_SAME_CARD}, source: ${src}), port ${SERVE_PORT}"
  local tty_flag=(-i); [[ -t 0 ]] && tty_flag=(-it)
  exec docker exec "${tty_flag[@]}" \
    -e "HOME=${WORK_DIR}" \
    -e "HF_HOME=${WORK_DIR}/.cache/huggingface" \
    -e "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}" \
    -e "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}" \
    -e "ASCEND_RT_VISIBLE_DEVICES=${devices}" \
    -w "${WORK_DIR}" \
    "${CONTAINER_NAME}" \
    bash -lc "vllm serve '${SERVE_MODEL}' --tensor-parallel-size ${tp} --host 0.0.0.0 --port ${SERVE_PORT} ${SERVE_EXTRA}"
}

cmd_check() {
  ensure_running
  local check_dev
  check_dev="$("${FIND_IDLE_NPU}" --pick 1)" || { echo "find_idle_npu.sh 选卡失败,check 用 die 0 兜底" >&2; check_dev=0; }
  echo "=== npu-smi info (inside, die ${check_dev}) ==="
  docker exec -e "ASCEND_RT_VISIBLE_DEVICES=${check_dev}" "${CONTAINER_NAME}" npu-smi info 2>&1 | sed -n '1,20p'
  echo "=== versions ==="
  docker exec "${CONTAINER_NAME}" bash -lc 'python -c "import vllm,vllm_ascend,torch,torch_npu; print(\"vllm\",vllm.__version__); print(\"vllm_ascend\",getattr(vllm_ascend,\"__version__\",\"?\")); print(\"torch\",torch.__version__); print(\"torch_npu\",torch_npu.__version__)"' 2>&1 | grep -vE "path string is NULL" | tail -8
  echo "=== import paths (image packages) ==="
  docker exec "${CONTAINER_NAME}" bash -lc 'python -c "import vllm,vllm_ascend; print(\"vllm:\",vllm.__file__); print(\"vllm_ascend:\",vllm_ascend.__file__)"' 2>&1 | grep -vE "path string is NULL" | tail -3
  echo "=== experiment bundle ==="
  docker exec "${CONTAINER_NAME}" test -f /workspace/pipeline_parallel/parallel_inference.py
  echo "/workspace/pipeline_parallel/parallel_inference.py: ok"
  echo "=== torch_npu op (npu sanity) ==="
  docker exec -e "ASCEND_RT_VISIBLE_DEVICES=${check_dev}" "${CONTAINER_NAME}" bash -lc \
    'python -c "import torch,torch_npu; x=torch.randn(8,8,device=\"npu\"); torch.npu.synchronize(); print(\"npu op ok\", float((x@x).sum().item()))"' 2>&1 | grep -vE "path string is NULL" | tail -3
}

main() {
  local cmd="${1:-}"; shift || true
  case "${cmd}" in
    start) cmd_start "$@" ;;
    restart) cmd_restart "$@" ;;
    stop) cmd_stop "$@" ;;
    serve) cmd_serve "$@" ;;
    check) cmd_check "$@" ;;
    shell|"") cmd_shell "$@" ;;
    logs) cmd_logs "$@" ;;
    -h|--help|help) usage; exit 0 ;;
    *) echo "Unknown command: ${cmd}" >&2; echo >&2; usage >&2; exit 1 ;;
  esac
}

main "$@"
