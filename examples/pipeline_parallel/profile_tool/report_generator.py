#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free SVG charts and Markdown/HTML report generation."""

from __future__ import annotations

import csv
import html
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


CHARTS = {
    "throughput_scaling.svg": (
        [("output_throughput", "Output tok/s", "#3478c0")],
        "Output throughput",
    ),
    "ttft_tpot_comparison.svg": (
        [
            ("ttft_p50_ms", "TTFT p50", "#3478c0"),
            ("tpot_p50_ms", "TPOT p50", "#e07a2d"),
        ],
        "TTFT and TPOT (ms)",
    ),
    "bubble_by_stage.svg": (
        [("bubble_ratio", "Bubble", "#b3436c")],
        "PP bubble ratio",
    ),
    "compute_memory_comm_share.svg": (
        [
            ("compute_share", "Compute", "#3b9c63"),
            ("memory_share", "Memory", "#e0ad2d"),
            ("communication_share", "Communication", "#8e5bc7"),
        ],
        "Kernel time share",
    ),
    "communication_breakdown.svg": (
        [
            ("pp_p2p_us", "PP Send/Recv", "#8e5bc7"),
            ("all_reduce_us", "AllReduce", "#3478c0"),
            ("all_gather_us", "AllGather", "#3b9c63"),
            ("reduce_scatter_us", "ReduceScatter", "#e0ad2d"),
            ("all_to_all_us", "AllToAll", "#b3436c"),
        ],
        "Communication time (us)",
    ),
    "scaling_efficiency.svg": (
        [("scaling_efficiency", "Efficiency", "#3b9c63")],
        "Scaling efficiency",
    ),
}


def _number(row: Mapping[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _svg_bar_chart(
    rows: Sequence[Mapping[str, Any]],
    series: Sequence[tuple[str, str, str]],
    title: str,
) -> str:
    width, height = 900, 420
    margin_left, margin_bottom, margin_top = 80, 70, 50
    chart_width = width - margin_left - 30
    chart_height = height - margin_top - margin_bottom
    values = [
        _number(row, key) for row in rows for key, _, _ in series
    ]
    maximum = max(values, default=1.0) or 1.0
    slot = chart_width / max(1, len(rows))
    bars: list[str] = []
    group_width = slot * 0.75
    bar_width = max(2.0, group_width / max(1, len(series)))
    for index, row in enumerate(rows):
        label = html.escape(str(row.get("case_id", index)))
        group_x = margin_left + index * slot + (slot - group_width) / 2
        for series_index, (key, _, color) in enumerate(series):
            value = _number(row, key)
            bar_height = chart_height * value / maximum
            x = group_x + series_index * bar_width
            y = margin_top + chart_height - bar_height
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_height:.1f}" fill="{color}"/>'
            )
        bars.append(
            f'<text x="{group_x + group_width / 2:.1f}" y="{height - 40}" '
            f'text-anchor="middle" font-size="13">{label}</text>'
        )
    legend = "".join(
        f'<rect x="{margin_left + index * 150}" y="{height - 18}" '
        f'width="12" height="12" fill="{color}"/>'
        f'<text x="{margin_left + index * 150 + 18}" y="{height - 7}" '
        f'font-size="12">{html.escape(label)}</text>'
        for index, (_, label, color) in enumerate(series)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="{width / 2}" y="28" text-anchor="middle" '
        f'font-size="20" font-family="sans-serif">{html.escape(title)}</text>'
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" '
        f'y2="{margin_top + chart_height}" stroke="#555"/>'
        f'<line x1="{margin_left}" y1="{margin_top + chart_height}" '
        f'x2="{margin_left + chart_width}" y2="{margin_top + chart_height}" '
        f'stroke="#555"/>{"".join(bars)}{legend}</svg>'
    )


def _recommendations(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    recommendations: list[str] = []
    if any(_number(row, "active_imbalance_ratio") >= 1.3 for row in rows):
        recommendations.append(
            "stage 活跃计算时间不均衡：建议按实测 layer cost 做带显存约束的 "
            "PP 加权分段，优先降低最慢 stage。"
        )
    if any(_number(row, "bubble_ratio") >= 0.4 for row in rows):
        recommendations.append(
            "PP 空泡率偏高：扫描并发度/微批大小，并分别比较 TTFT 与 TPOT，"
            "避免只追求吞吐。"
        )
    if any(_number(row, "communication_share") >= 0.3 for row in rows):
        recommendations.append(
            "通信占比偏高：保持 TP/EP group 的拓扑局部性，检查 PP stage "
            "边界与 Send/Recv、AllToAll 的计算重叠。"
        )
    if any(_number(row, "memory_share") >= 0.35 for row in rows):
        recommendations.append(
            "内存访问占比偏高：检查算子融合、张量布局、权重/KV 访问以及"
            "热路径 CPU-NPU 同步。"
        )
    if not recommendations:
        recommendations.append(
            "当前轻量数据未触发明确瓶颈阈值；应结合代表点多 rank trace 再判断。"
        )
    return recommendations


def _derive_scaling_efficiency(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    copied = [dict(row) for row in rows]
    if not copied:
        return copied
    baseline = _number(copied[0], "output_throughput")
    baseline_world = max(
        1, int(_number(copied[0], "pp") * _number(copied[0], "tp"))
    )
    for row in copied:
        world = max(1, int(_number(row, "pp") * _number(row, "tp")))
        ideal_scale = world / baseline_world
        actual_scale = (
            _number(row, "output_throughput") / baseline if baseline else 0
        )
        row["scaling_efficiency"] = (
            actual_scale / ideal_scale if ideal_scale else 0
        )
    return copied


def _aggregate_for_charts(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("case_id", "")), []).append(row)
    metric_keys = {
        key
        for series, _ in CHARTS.values()
        for key, _, _ in series
        if key != "scaling_efficiency"
    }
    aggregated: list[dict[str, Any]] = []
    for case_id, case_rows in groups.items():
        first = case_rows[0]
        item: dict[str, Any] = {
            "case_id": case_id,
            "pp": first.get("pp", ""),
            "tp": first.get("tp", ""),
            "ep": first.get("ep", ""),
        }
        for key in metric_keys:
            item[key] = statistics.median(
                _number(row, key) for row in case_rows
            )
        aggregated.append(item)
    return _derive_scaling_efficiency(aggregated)


def generate_report(
    output_dir: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any],
) -> dict[str, str]:
    output = Path(output_dir)
    charts_dir = output / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    derived = _derive_scaling_efficiency(rows)
    chart_rows = _aggregate_for_charts(derived)
    fieldnames = sorted({key for row in derived for key in row})
    with (output / "summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(derived)
    for filename, (series, title) in CHARTS.items():
        (charts_dir / filename).write_text(
            _svg_bar_chart(chart_rows, series, title), encoding="utf-8"
        )
    recommendations = _recommendations(chart_rows)
    table_header = (
        "| Case | PP | TP | EP | Throughput | TTFT p50 (ms) | "
        "TPOT p50 (ms) | Bubble |\n"
        "|---|---:|---:|:---:|---:|---:|---:|---:|"
    )
    table_rows = [
        "| {case_id} | {pp} | {tp} | {ep} | {throughput:.3f} | "
        "{ttft:.3f} | {tpot:.3f} | {bubble:.3f} |".format(
            case_id=row.get("case_id", ""),
            pp=row.get("pp", ""),
            tp=row.get("tp", ""),
            ep=row.get("ep", ""),
            throughput=_number(row, "output_throughput"),
            ttft=_number(row, "ttft_p50_ms"),
            tpot=_number(row, "tpot_p50_ms"),
            bubble=_number(row, "bubble_ratio"),
        )
        for row in chart_rows
    ]
    chart_links = "\n".join(
        f"![{title}](charts/{filename})"
        for filename, (_, title) in CHARTS.items()
    )
    markdown = (
        "# vLLM-Ascend 二维并行性能边界报告\n\n"
        f"- Run ID: `{metadata.get('run_id', '')}`\n"
        f"- Model: `{metadata.get('model', '')}`\n"
        "- TTFT/TPOT 均来自真实 SSE 流式 token 时间戳。\n\n"
        "## 结果摘要\n\n"
        f"{table_header}\n" + "\n".join(table_rows) + "\n\n"
        "## 对比图\n\n"
        f"{chart_links}\n\n"
        "## PP 流水线优化建议\n\n"
        + "\n".join(f"- {item}" for item in recommendations)
        + "\n\n"
        "## 口径与限制\n\n"
        "- 计算/访存/通信占比首先是 kernel 执行时间分类占比；若 profiler "
        "缺少 PMU/HBM 字段，不将其宣称为硬件利用率。\n"
        "- Profiling 只采代表点和边界点，轻量 benchmark 数据不与 profiling "
        "扰动后的数据混用。\n"
    )
    report_md = output / "report.md"
    report_md.write_text(markdown, encoding="utf-8")
    report_html = output / "report.html"
    report_html.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>vLLM-Ascend parallel benchmark</title>"
        "<style>body{font-family:Segoe UI,sans-serif;max-width:1100px;"
        "margin:2rem auto;line-height:1.6}img{max-width:100%;}"
        "pre{white-space:pre-wrap}</style></head><body>"
        "<h1>vLLM-Ascend 二维并行性能边界报告</h1>"
        f"<pre>{html.escape(markdown)}</pre>"
        + "".join(
            f"<h2>{html.escape(title)}</h2>"
            f"<img src='charts/{filename}' alt='{html.escape(title)}'>"
            for filename, (_, title) in CHARTS.items()
        )
        + "</body></html>",
        encoding="utf-8",
    )
    (output / "report_metadata.json").write_text(
        json.dumps(
            {
                "metadata": dict(metadata),
                "recommendations": recommendations,
                "rows": len(derived),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"report_md": str(report_md), "report_html": str(report_html)}
