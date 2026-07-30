#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Analyze every rank produced by torch_npu profiler and write an index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def analyze_profiles(profile_root: Path) -> dict[str, object]:
    from torch_npu.profiler.profiler import analyse

    profiles = sorted(profile_root.glob("**/*ascend_pt"))
    if not profiles:
        raise RuntimeError(f"no *ascend_pt directories under {profile_root}")
    results: list[dict[str, str]] = []
    for rank_index, profile in enumerate(profiles):
        analyse(str(profile))
        output = profile / "ASCEND_PROFILER_OUTPUT"
        trace = output / "trace_view.json"
        if not trace.is_file():
            raise RuntimeError(f"trace_view.json missing after analyse: {profile}")
        results.append(
            {
                "rank_index": str(rank_index),
                "profile": str(profile),
                "trace_view": str(trace),
                "kernel_details": str(output / "kernel_details.csv"),
                "operator_details": str(output / "operator_details.csv"),
            }
        )
    index = {"profile_root": str(profile_root), "ranks": results}
    (profile_root / "profile_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_profiles(args.profile_root)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
