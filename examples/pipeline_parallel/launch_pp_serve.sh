#!/usr/bin/env bash
# Backward-compatible PP=2, TP=1 wrapper.
# Use parallel_inference.py directly for PP×TP/EP or PP+weight offload.
set -euo pipefail

MODEL="${1:?usage: $0 <model_path> <die1,die2> [port] [extra vllm args...]}"
DIES="${2:?die list, e.g. 2,3 (same card) or 2,6 (cross card)}"

if [[ $# -ge 3 && "$3" =~ ^[0-9]+$ ]]; then
  PORT="$3"
  shift 3
else
  PORT=8000
  shift 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/parallel_inference.py" \
  --model "$MODEL" \
  --devices "$DIES" \
  --tp 1 \
  --pp 2 \
  --port "$PORT" \
  -- "$@"
