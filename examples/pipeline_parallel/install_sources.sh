#!/bin/bash
# install_sources.sh — 在 vllm-ascend 容器内从源码安装 vllm + vllm-ascend(编出 C 扩展)
#
# 背景:docker.sh 把宿主源码 bind-mount 进 /vllm-workspace/{vllm,vllm-ascend},
#   遮住了镜像里预编的 *.so -> vllm_ascend_C(vllm._C 同理)import 失败,
#   vllm serve 在 model forward(layernorm -> enable_custom_op)处 ERR99999。
#   本脚本在容器内对两仓库跑 editable 安装,把编出来的 .so 落回宿主挂载目录(持久、抗 recreate)。
#
# 关键事实(已实测):
#   - 容器 env 已带 SOC_VERSION=ascend910_9391 / ASCEND_HOME_PATH / LD_LIBRARY_PATH / PATH,
#     docker exec 直接继承,无需再 source set_env.sh。
#   - 工具链:cmake 4.4 / gcc / pybind11 3.0.4 / torch 2.10.0+cpu。pip 源可达(huggingface.co 才被墙)。
#   - vllm 是 +empty 无设备构建:setup.py _no_device() -> ext_modules=[] 不产 vllm._C,Ascend 不需要它。
#     Dockerfile.a3.openEuler line 51 全程未装 rust 工具链却构建成功,证明 empty 路径不触发 cargo。
#     故两仓库都用 VLLM_TARGET_DEVICE=empty / 默认 走 `pip install -e . --no-build-isolation`,
#     命令与官方 Dockerfile line 51/67 一致,只是落到 bind-mount 的宿主目录。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-vllm_dynamic}"
# 容器内源码路径(docker.sh 的 bind mount 目标)
VLLM_DIR="${VLLM_DIR:-/vllm-workspace/vllm}"
VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/vllm-workspace/vllm-ascend}"
MAX_JOBS="${MAX_JOBS:-4}"            # cmake/编译并行度;容器 nproc=1,适当抬一点
SKIP_VLLM_BUILD="${SKIP_VLLM_BUILD:-0}"  # =1 强制跳过 vllm 全量构建,只重生 _version.py

log() { printf '\033[1;34m[install_sources]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[install_sources ERROR]\033[0m %s\n' "$*" >&2; }

container_running() { docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; }

ensure_container() {
  if container_running; then return 0; fi
  log "容器 '${CONTAINER_NAME}' 未运行,先 ./docker.sh start"
  "${SCRIPT_DIR}/docker.sh" start
  container_running || { err "容器仍不可用"; exit 1; }
}

# 容器内构建脚本:通过 stdin 喂给 docker exec,避免宿主/容器路径与引号纠缠。
inner_build() {
  # build.sh 的 CPU_NUM 取 /proc/cpuinfo*2(=1280),不受 OMP_NUM_THREADS 影响;
  # 镜像默认 OMP_NUM_THREADS=1 让每个进程 1 OPM 线程,配合高 -j 反而避免过订,故不动 OMP。
  # MAX_JOBS 向下 cap CPU_NUM:共享机做公民,卡到 64(宿主 640 核、load~48)。
  docker exec -i -e "MAX_JOBS=${MAX_JOBS}" "${CONTAINER_NAME}" bash -euo pipefail
}

run_build() {
  inner_build <<EOF
set -x
# 非 login shell 的 LD_LIBRARY_PATH 不含 cann-9.0.1/aarch64-linux/lib64(放 libhccl.so),
# cmake configure 里 import torch_npu 会报 libhccl.so 缺失 -> CMake configuration failed。
# 照 Dockerfile.a3.openEuler line 65-66 显式 source 两份 set_env.sh。
set +u  # set_env.sh 里引用了未定义变量(如 ZSH_VERSION),-u 下会报错
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
python -c "import torch_npu; print('torch_npu OK (libhccl resolved)')"

echo "=== build deps ==="
pip install --no-cache-dir -q setuptools_rust setuptools_scm wheel

echo "=== 1/2 vllm-ascend: pip install -e . (cmake builds vllm_ascend_C) ==="
cd "${VLLM_ASCEND_DIR}"
# 重生 _version.py(setuptools-scm 产物不进 git,checkout 覆盖后会丢)
python -c "from setuptools_scm import get_version as g; print(g(write_to='vllm_ascend/_version.py'))" || true
pip install -e . --no-build-isolation -v
python -c "import vllm_ascend.vllm_ascend_C; print('vllm_ascend_C OK ->', vllm_ascend.vllm_ascend_C.__file__)"

echo "=== 2/2 vllm (VLLM_TARGET_DEVICE=empty,无设备构建,不产 _C) ==="
cd "${VLLM_DIR}"
python -c "from setuptools_scm import get_version as g; print(g(write_to='vllm/_version.py'))" || true
# 照 Dockerfile.a3.openEuler line 51:VLLM_TARGET_DEVICE=empty -> _no_device() -> ext_modules=[] 不编 _C。
# 该 Dockerfile 全程未装 rust 工具链却构建成功,证明 empty 构建路径不触发 cargo(setuptools_rust 的 optional ext 跳过)。
VLLM_TARGET_DEVICE=empty pip install -e . --no-build-isolation --no-deps -v
python -c "import vllm; print('vllm', vllm.__version__, '->', vllm.__file__)"
EOF
}

main() {
  local cmd="${1:-build}"
  case "${cmd}" in
    build|"")
      log "容器: ${CONTAINER_NAME}"
      log "MAX_JOBS=${MAX_JOBS}  vllm=${VLLM_DIR}  vllm-ascend=${VLLM_ASCEND_DIR}"
      ensure_container
      if [[ "${SKIP_VLLM_BUILD}" == "1" ]]; then
        log "SKIP_VLLM_BUILD=1 -> 仅装 vllm-ascend(注:此处仍跑全脚本,vllm 段会被 cargo 探测自动跳过)"
      fi
      run_build
      log "完成。自检:SERVE_DEVICES=6,7 ./docker.sh check;  起服务:SERVE_MODEL=<本地快照路径> ./docker.sh serve"
      ;;
    check)
      ensure_container
      docker exec "${CONTAINER_NAME}" bash -lc \
        'python -c "import vllm,vllm_ascend,vllm_ascend.vllm_ascend_C,torch,torch_npu; print(\"vllm\",vllm.__version__,\"| vllm_ascend_C OK | torch\",torch.__version__)"' \
        2>&1 | grep -vE "path string is NULL"
      ;;
    -h|--help|help)
      sed -n '2,30p' "${BASH_SOURCE[0]}"
      ;;
    *) err "未知命令: ${cmd}"; echo "用法: $0 [build|check]" >&2; exit 1 ;;
  esac
}

main "$@"
