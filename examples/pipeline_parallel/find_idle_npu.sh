#!/usr/bin/env bash
set -euo pipefail

# Die-level idle selector for Ascend. Always strict (filter out dies with running proc-mem process).
#
# Usage:
#   ./find_idle_npu.sh [--pick [N]|--list] [--quiet] [--wait] [--wait-seconds N] [max_aicore] [max_hbm_pct]
#
# Examples:
#   ./find_idle_npu.sh --pick 2
#   ./find_idle_npu.sh --list --quiet 5 10
#   ./find_idle_npu.sh --pick 2 --wait --wait-seconds 30
#   ./find_idle_npu.sh --pick 2 --same-card        # 强制同卡双 die(走 HCCS,避免跨卡 HCCL 走网络面)

MODE="pick"
PICK_N=1
QUIET=0
WAIT=0
WAIT_SECONDS=60
FORCE_SAME_CARD="${FORCE_SAME_CARD:-0}"
POSITIONAL=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pick)
      MODE="pick"
      if [[ "${2:-}" =~ ^[0-9]+$ ]]; then
        PICK_N="$2"
        shift 2
      else
        shift
      fi
      ;;
    --list)
      MODE="${1#--}"; shift ;;
    --quiet)
      QUIET=1; shift ;;
    --wait)
      WAIT=1; shift ;;
    --wait-seconds)
      if [[ "${2:-}" =~ ^[0-9]+$ ]] && [[ "${2:-0}" -gt 0 ]]; then
        WAIT_SECONDS="$2"
        shift 2
      else
        echo "Invalid --wait-seconds value: ${2:-}" >&2
        exit 1
      fi
      ;;
    --same-card)
      FORCE_SAME_CARD=1; shift ;;
    -h|--help)
      sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    [0-9]*)
      POSITIONAL+=("$1"); shift ;;
    *)
      echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

MAX_AICORE="${POSITIONAL[0]:-5}"
MAX_HBM_PCT="${POSITIONAL[1]:-10}"

log(){ [[ "$QUIET" -eq 1 ]] || echo "$*" >&2; }
command -v npu-smi >/dev/null 2>&1 || { log "ERROR: npu-smi not found"; exit 2; }

query_once() {
RESULT="$({
  export MAX_AICORE MAX_HBM_PCT MODE PICK_N FORCE_SAME_CARD
  python3 - <<'PY'
import os, re, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

max_aicore = int(os.environ["MAX_AICORE"])
max_hbm = int(os.environ["MAX_HBM_PCT"])
mode = os.environ["MODE"]
pick_n = int(os.environ.get("PICK_N", "1"))
force_same_card = os.environ.get("FORCE_SAME_CARD", "0") == "1"

def run(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (p.stdout or "") + (p.stderr or "")

def mapping():
    text = run("npu-smi info -m")
    rows = []
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d-]+)\s+(\S+)\s*$", line)
        if not m:
            continue
        card, chip, logic, _, name = m.groups()
        if name.lower().startswith("ascend"):
            rows.append((int(card), int(chip), int(logic)))
    return sorted(rows, key=lambda x: x[2])

def die_metrics(usages_text):
    out, cur = {}, {}
    def flush():
        nonlocal cur
        if "chip" in cur:
            out[cur["chip"]] = (cur.get("aicore", 0), cur.get("hbm", cur.get("ddr", 0)))
        cur = {}
    for line in usages_text.splitlines():
        if ":" not in line:
            continue
        k, v = [x.strip().lower() for x in line.split(":", 1)]
        if k == "chip id":
            flush()
            if v.isdigit():
                cur["chip"] = int(v)
            continue
        if not v.isdigit():
            continue
        n = int(v)
        if "aicore usage rate" in k:
            cur["aicore"] = n
        elif "hbm usage rate" in k:
            cur["hbm"] = n
        elif "ddr usage rate" in k and "hugepages" not in k:
            cur["ddr"] = n
    flush()
    return out

def has_proc(card, chip):
    t = run(f"npu-smi info -t proc-mem -i {card} -c {chip}")
    if re.search(r"not support|not available|error", t, re.I):
        return False
    for line in t.splitlines():
        if re.search(r"\bPID\b", line, re.I) and re.search(r":\s*\d+\b", line):
            return True
        if re.search(r"Process\s+Name", line, re.I) and ":" in line:
            val = line.split(":",1)[1].strip()
            if val and val not in ("-","N/A","None"):
                return True
    return False

def select_for_pick(candidates, n, force_same_card=False):
    """Pick n dies, preferring same-card pairs when n is even.
    When force_same_card=True, ONLY return same-card pairs (no cross-card
    fallback) so multi-die TP/PP uses HCCS instead of the network plane."""
    if n <= 0 or not candidates:
        return []
    if n % 2 == 1:
        return candidates[:n]

    by_card = {}
    for x in candidates:
        by_card.setdefault(x["card_id"], []).append(x)

    selected = []
    used_logic = set()
    for card_id in sorted(by_card.keys()):
        group = sorted(by_card[card_id], key=lambda x: x["logic_id"])
        while len(group) >= 2 and len(selected) + 2 <= n:
            a = group.pop(0)
            b = group.pop(0)
            selected.extend([a, b])
            used_logic.add(a["logic_id"])
            used_logic.add(b["logic_id"])
        if len(selected) >= n:
            break

    # 跨卡兜底:force_same_card 时跳过,凑不齐同卡对就返回少于 n(让上层 --wait 等待)
    if not force_same_card and len(selected) < n:
        for x in candidates:
            if x["logic_id"] in used_logic:
                continue
            selected.append(x)
            if len(selected) >= n:
                break

    return sorted(selected, key=lambda x: x["logic_id"])

mp = mapping()
if not mp:
    print("ERR=no_mapping")
    raise SystemExit

cards = sorted(set(c for c,_,_ in mp))
card_m = {}
for c in cards:
    t = run(f"npu-smi info -t usages -i {c}")
    if re.search(r"failed|initialize failed", t, re.I):
        continue
    card_m[c] = die_metrics(t)

base = []
for card, chip, logic in mp:
    m = card_m.get(card, {}).get(chip)
    if not m:
        continue
    aicore, hbm = m
    if aicore > max_aicore or hbm > max_hbm:
        continue
    base.append({"logic_id": logic, "card_id": card, "chip_id": chip, "aicore": aicore, "hbm_pct": hbm})

base.sort(key=lambda x: x["logic_id"])

idle = []
workers = min(8, max(1, len(base)))
with ThreadPoolExecutor(max_workers=workers) as ex:
    fut_map = {ex.submit(has_proc, x["card_id"], x["chip_id"]): x for x in base}
    for fut in as_completed(fut_map):
        x = fut_map[fut]
        if not fut.result():
            idle.append(x)
idle.sort(key=lambda x: x["logic_id"])

if mode == "pick":
    idle = select_for_pick(idle, pick_n, force_same_card)

pick = idle[0]["logic_id"] if idle else None
ids = " ".join(str(x["logic_id"]) for x in idle)
print(f"PICK={'' if pick is None else pick}")
print(f"IDS={ids}")
PY
} )"
}

while true; do
  query_once
  if [[ "$RESULT" == *"ERR=no_mapping"* ]] || [[ -z "$RESULT" ]]; then
    log "ERROR: no NPU mapping from npu-smi info -m"
    exit 4
  fi

  PICK="$(printf '%s\n' "$RESULT" | sed -n 's/^PICK=//p')"
  IDS="$(printf '%s\n' "$RESULT" | sed -n 's/^IDS=//p')"
  if [[ -n "$IDS" ]]; then
    break
  fi

  if [[ "$WAIT" -ne 1 ]]; then
    if [[ "$FORCE_SAME_CARD" == "1" ]]; then
      log "No same-card idle die pair (strict, same-card forced)"
    else
      log "No idle die (aicore<=${MAX_AICORE}% hbm<=${MAX_HBM_PCT}%, strict)"
    fi
    exit 5
  fi

  if [[ "$FORCE_SAME_CARD" == "1" ]]; then
    log "No same-card idle die pair, waiting ${WAIT_SECONDS}s and retrying..."
  else
    log "No idle die, waiting ${WAIT_SECONDS}s and retrying..."
  fi
  sleep "$WAIT_SECONDS"
done

case "$MODE" in
  pick)
    if [[ "$PICK_N" -le 1 ]]; then
      echo "$PICK"
    else
      echo "$IDS" | tr ' ' ','
    fi
    ;;
  list)
    echo "$IDS"
    ;;
esac
