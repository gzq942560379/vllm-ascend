#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible streaming workload generator with true token timings."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


def percentile(values: Sequence[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True)
class RequestTimeline:
    started_at: float
    token_timestamps: tuple[float, ...]
    completed_at: float
    prompt_tokens: int
    output_tokens: int

    def metrics(self) -> dict[str, Any]:
        first = self.token_timestamps[0] if self.token_timestamps else None
        ttft = None if first is None else (first - self.started_at) * 1000.0
        tpot = None
        if first is not None and self.output_tokens > 1:
            tpot = (
                (self.completed_at - first) * 1000.0 / (self.output_tokens - 1)
            )
        intervals = [
            (right - left) * 1000.0
            for left, right in zip(
                self.token_timestamps, self.token_timestamps[1:]
            )
        ]
        return {
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "itl_p50_ms": percentile(intervals, 50),
            "itl_p95_ms": percentile(intervals, 95),
            "itl_p99_ms": percentile(intervals, 99),
            "e2e_ms": (self.completed_at - self.started_at) * 1000.0,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
        }


def _extract_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if isinstance(choice.get("text"), str):
        return choice["text"]
    delta = choice.get("delta") or {}
    return str(delta.get("content") or "")


def stream_one_request(
    *,
    url: str,
    model: str,
    prompt: str,
    output_tokens: int,
    request_timeout_seconds: int,
    request_id: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    timestamps: list[float] = []
    usage: dict[str, Any] = {}
    text_parts: list[str] = []
    try:
        with urllib.request.urlopen(
            request, timeout=request_timeout_seconds
        ) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                text = _extract_text(event)
                if text:
                    text_parts.append(text)
                choices = event.get("choices") or []
                token_ids = choices[0].get("token_ids") if choices else None
                arrived = time.perf_counter()
                if token_ids:
                    timestamps.extend(arrived for _ in token_ids)
                elif text:
                    timestamps.append(arrived)
        completed = time.perf_counter()
        emitted = int(usage.get("completion_tokens") or len(timestamps))
        timeline = RequestTimeline(
            started_at=started,
            token_timestamps=tuple(timestamps),
            completed_at=completed,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=emitted,
        )
        return {
            "request_id": request_id,
            "ok": True,
            **timeline.metrics(),
            "started_at_monotonic": started,
            "completed_at_monotonic": completed,
            "text": "".join(text_parts),
        }
    except (OSError, urllib.error.URLError, ValueError) as exc:
        return {
            "request_id": request_id,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "e2e_ms": (time.perf_counter() - started) * 1000.0,
            "started_at_monotonic": started,
            "completed_at_monotonic": time.perf_counter(),
        }


def build_prompt(target_tokens: int) -> str:
    # The API usage field is authoritative. This deterministic prompt is long
    # enough for comparison and avoids random synthetic token IDs.
    unit = "请分析昇腾大模型推理中流水线并行的计算、通信与内存开销。"
    repetitions = max(1, target_tokens // 18)
    return (unit * repetitions)[: max(32, target_tokens * 3)]


def tokenize_count(
    *,
    url: str,
    model: str,
    prompt: str,
    timeout_seconds: int,
) -> int:
    request = urllib.request.Request(
        f"{url.rstrip('/')}/tokenize",
        data=json.dumps({"model": model, "prompt": prompt}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return int(payload["count"])


def build_exact_prompt(
    target_tokens: int,
    *,
    count_tokens: Callable[[str], int],
) -> str:
    """Construct meaningful text whose tokenizer count exactly matches target."""
    if target_tokens < 1:
        raise ValueError("target_tokens must be positive")
    seed = (
        "请分析昇腾大模型推理中流水线并行的计算、通信、显存访问与调度空泡，"
        "并给出可以验证的优化依据。"
    )
    seed_count = count_tokens(seed)
    if seed_count == target_tokens:
        return seed
    if seed_count > target_tokens:
        lower, upper = 1, len(seed)
        best = seed[:1]
        while lower <= upper:
            middle = (lower + upper) // 2
            candidate = seed[:middle]
            count = count_tokens(candidate)
            if count <= target_tokens:
                best = candidate
                lower = middle + 1
            else:
                upper = middle - 1
        seed = best

    unit = " 流水线并行性能测试。"
    lower, upper = 0, max(1, target_tokens * 2)
    best = seed
    best_count = count_tokens(best)
    while lower <= upper:
        middle = (lower + upper) // 2
        candidate = seed + unit * middle
        count = count_tokens(candidate)
        if count <= target_tokens:
            if count >= best_count:
                best, best_count = candidate, count
            lower = middle + 1
        else:
            upper = middle - 1
    if best_count == target_tokens:
        return best

    fillers = ("。", " 的", " a", " 1", "\n")
    for _ in range(target_tokens * 2):
        improvement: tuple[int, str] | None = None
        for filler in fillers:
            candidate = best + filler
            count = count_tokens(candidate)
            if best_count < count <= target_tokens:
                if improvement is None or count > improvement[0]:
                    improvement = (count, candidate)
        if improvement is None:
            break
        best_count, best = improvement
        if best_count == target_tokens:
            return best
    raise RuntimeError(
        f"could not construct exactly {target_tokens} tokens; "
        f"closest safe prompt has {best_count}"
    )


def run_workload(
    *,
    url: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    num_requests: int,
    concurrency: int,
    timeout_seconds: int,
    prompt: str | None = None,
) -> list[dict[str, Any]]:
    prompt = prompt if prompt is not None else build_prompt(input_tokens)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency,
        thread_name_prefix="vllm-stream",
    ) as executor:
        futures = [
            executor.submit(
                stream_one_request,
                url=url,
                model=model,
                prompt=prompt,
                output_tokens=output_tokens,
                request_timeout_seconds=timeout_seconds,
                request_id=index,
            )
            for index in range(num_requests)
        ]
        return [future.result() for future in futures]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--num-requests", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = run_workload(
        url=args.url,
        model=args.model,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        timeout_seconds=args.timeout_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    failed = sum(not row["ok"] for row in rows)
    print(json.dumps({"requests": len(rows), "failed": failed}))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
