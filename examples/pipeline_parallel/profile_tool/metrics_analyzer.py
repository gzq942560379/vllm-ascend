#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Post-process CANN profiler CSV files into parallel performance metrics."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


COMMUNICATION_PATTERNS = {
    "all_reduce": re.compile(r"all.?reduce", re.I),
    "all_gather": re.compile(r"all.?gather", re.I),
    "reduce_scatter": re.compile(r"reduce.?scatter", re.I),
    "all_to_all": re.compile(r"all.?to.?all|alltoall|dispatch|combine", re.I),
    "send_recv": re.compile(r"\b(send|recv|receive)\b|hcomsend|hcomreceive", re.I),
}
MEMORY_PATTERN = re.compile(
    r"copy|memcpy|memset|transpose|reshape|view|slice|index|gather|scatter",
    re.I,
)
COMPUTE_PATTERN = re.compile(
    r"matmul|gemm|mm\b|conv|attention|flash|softmax|norm|add|mul|"
    r"activation|silu|gelu|moe|expert",
    re.I,
)


def communication_kind(name: str) -> str | None:
    for kind, pattern in COMMUNICATION_PATTERNS.items():
        if pattern.search(name):
            return kind
    if re.search(r"\bhcom|hccl\b", name, re.I):
        return "other_collective"
    return None


def classify_kernel(name: str) -> str:
    if communication_kind(name):
        return "communication"
    if MEMORY_PATTERN.search(name):
        return "memory"
    if COMPUTE_PATTERN.search(name):
        return "compute"
    return "other"


def _duration_us(row: Mapping[str, Any]) -> float:
    preferred = (
        "Duration(us)",
        "Duration (us)",
        "Task Duration(us)",
        "duration_us",
        "Duration",
    )
    for key in preferred:
        if key in row and str(row[key]).strip():
            try:
                return float(str(row[key]).replace(",", ""))
            except ValueError:
                pass
    return 0.0


def _start_us(row: Mapping[str, Any]) -> float | None:
    for key in (
        "Start Time(us)",
        "Start Time (us)",
        "Task Start Time(us)",
        "start_us",
    ):
        if key in row and str(row[key]).strip():
            try:
                return float(str(row[key]).replace(",", ""))
            except ValueError:
                pass
    return None


def _name(row: Mapping[str, Any]) -> str:
    for key in ("Name", "Op Name", "Task Name", "name"):
        if key in row:
            return str(row[key])
    return ""


def _percentile(values: Sequence[float], value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * value / 100
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_kernel_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    duration = defaultdict(float)
    communication_values: dict[str, list[float]] = defaultdict(list)
    fields: set[str] = set()
    earliest_start: float | None = None
    latest_end: float | None = None
    for row in rows:
        fields.update(row.keys())
        name = _name(row)
        value = _duration_us(row)
        start = _start_us(row)
        if start is not None:
            earliest_start = (
                start if earliest_start is None else min(earliest_start, start)
            )
            end = start + value
            latest_end = end if latest_end is None else max(latest_end, end)
        category = classify_kernel(name)
        duration[category] += value
        kind = communication_kind(name)
        if kind:
            communication_values[kind].append(value)
    total = sum(duration.values())
    categories = ("compute", "memory", "communication", "other")
    communication = {
        kind: {
            "count": len(values),
            "total_us": sum(values),
            "p50_us": _percentile(values, 50),
            "p95_us": _percentile(values, 95),
            "max_us": max(values),
        }
        for kind, values in sorted(communication_values.items())
    }
    hardware_fields = [
        field
        for field in sorted(fields)
        if re.search(r"aicore|cube|hbm|llc|bandwidth|pmu", field, re.I)
    ]
    return {
        "duration_us": {key: duration[key] for key in categories},
        "share": {
            key: duration[key] / total if total else 0.0 for key in categories
        },
        "communication": communication,
        "hardware_metrics_available": bool(hardware_fields),
        "hardware_metric_fields": hardware_fields,
        "classification_version": 1,
        "measurement_makespan_us": (
            latest_end - earliest_start
            if earliest_start is not None and latest_end is not None
            else None
        ),
    }


def analyze_kernel_csv(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return summarize_kernel_rows(csv.DictReader(handle))


def analyze_bubbles(
    stages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    active_values: list[float] = []
    for stage in stages:
        active = float(stage["active_compute_us"])
        makespan = float(stage["makespan_us"])
        communication = float(stage.get("communication_us", 0.0))
        bubble = 1.0 - active / makespan if makespan else 0.0
        results.append(
            {
                **dict(stage),
                "active_compute_us": active,
                "communication_us": communication,
                "makespan_us": makespan,
                "bubble_ratio": max(0.0, min(1.0, bubble)),
            }
        )
        active_values.append(active)
    positive = [value for value in active_values if value > 0]
    imbalance = max(positive) / min(positive) if positive else 0.0
    return {"stages": results, "active_imbalance_ratio": imbalance}


def suggest_contiguous_stage_boundaries(
    layer_costs: Sequence[float],
    *,
    stages: int,
) -> dict[str, Any]:
    """Minimize the maximum contiguous PP stage cost.

    This is an advisory partition only. Embedding/lm_head and memory constraints
    can be represented as additional costs before calling this function.
    """
    costs = [float(value) for value in layer_costs]
    if not costs or any(value < 0 for value in costs):
        raise ValueError("layer_costs must contain non-negative values")
    if not 1 <= stages <= len(costs):
        raise ValueError("stages must be in [1, number of layers]")

    prefix = [0.0]
    for value in costs:
        prefix.append(prefix[-1] + value)
    count = len(costs)
    infinity = float("inf")
    best = [[infinity] * (stages + 1) for _ in range(count + 1)]
    split = [[0] * (stages + 1) for _ in range(count + 1)]
    best[0][0] = 0.0
    for end in range(1, count + 1):
        for stage_count in range(1, min(stages, end) + 1):
            for start in range(stage_count - 1, end):
                candidate = max(
                    best[start][stage_count - 1],
                    prefix[end] - prefix[start],
                )
                if candidate < best[end][stage_count]:
                    best[end][stage_count] = candidate
                    split[end][stage_count] = start
    boundaries = [count]
    end = count
    stage_count = stages
    while stage_count:
        end = split[end][stage_count]
        boundaries.append(end)
        stage_count -= 1
    boundaries.reverse()
    stage_costs = [
        prefix[right] - prefix[left]
        for left, right in zip(boundaries, boundaries[1:])
    ]
    return {
        "boundaries": boundaries,
        "stage_costs": stage_costs,
        "max_stage_cost": max(stage_costs),
        "imbalance_ratio": (
            max(stage_costs) / min(stage_costs)
            if min(stage_costs) > 0
            else float("inf")
        ),
    }
