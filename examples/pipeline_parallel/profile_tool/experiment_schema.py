#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Experiment schema, matrix expansion, and vLLM CLI capability adaptation."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


DEFAULT_MODEL_PATH = "/models/Qwen3-30B-A3B"
MAX_ASCEND_DIES = 16


@dataclass(frozen=True)
class ParallelCase:
    case_id: str
    pp: int
    tp: int
    ep: bool = False
    cp: int = 1
    profile: bool = False

    @property
    def world_size(self) -> int:
        # EP changes expert group construction; it does not multiply processes.
        return self.pp * self.tp * self.cp

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "pp": self.pp,
            "tp": self.tp,
            "ep": self.ep,
            "cp": self.cp,
            "profile": self.profile,
            "world_size": self.world_size,
        }


@dataclass(frozen=True)
class CapabilitySet:
    help_text: str
    expert_parallel_flag: str | None
    context_parallel_flag: str | None
    profiler_config: bool
    kv_cache_memory_bytes: bool

    @classmethod
    def from_help_text(cls, text: str) -> "CapabilitySet":
        lowered = text.lower()
        ep_flag = (
            "--enable-expert-parallel"
            if "--enable-expert-parallel" in lowered
            else None
        )
        cp_candidates = (
            "--context-parallel-size",
            "--decode-context-parallel-size",
        )
        cp_flag = next((flag for flag in cp_candidates if flag in lowered), None)
        return cls(
            help_text=text,
            expert_parallel_flag=ep_flag,
            context_parallel_flag=cp_flag,
            profiler_config="--profiler-config" in lowered,
            kv_cache_memory_bytes="--kv-cache-memory-bytes" in lowered,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "expert_parallel_flag": self.expert_parallel_flag,
            "context_parallel_flag": self.context_parallel_flag,
            "profiler_config": self.profiler_config,
            "kv_cache_memory_bytes": self.kv_cache_memory_bytes,
        }


@dataclass(frozen=True)
class FlagDecision:
    flags: tuple[str, ...]
    status: str | None = None
    reason: str = ""


def default_spec() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "server": {
            "host": "192.168.13.190",
            "ssh_user": "root",
            "remote_root": "/home/vllm/l00977701/runtime/parallel_bench_runs",
        },
        "container": {
            "name": "qwen3_parallel_nightly",
            "expected_vllm_version": "",
            "expected_vllm_ascend_version": "",
        },
        "model": {
            "name": "Qwen3-30B-A3B",
            "path": DEFAULT_MODEL_PATH,
            "kind": "moe",
            "served_name": "qwen3-30b-a3b-parallel-bench",
            "max_model_len": 4096,
        },
        "matrix": {"preset": "quick", "cases": []},
        "workload": {
            "input_tokens": [128, 512, 2048],
            "output_tokens": [64, 256],
            "concurrency": [1, 4, 16, 32],
            "warmup_requests": 4,
            "repetitions": 3,
            "requests_per_point": 64,
            "temperature": 0,
            "ignore_eos": True,
            "request_timeout_seconds": 600,
        },
        "profiling": {
            "input_tokens": 512,
            "output_tokens": 64,
            "num_requests": 16,
            "concurrency": 4,
        },
        "resource": {
            "max_dies": MAX_ASCEND_DIES,
            "poll_seconds": 60,
            "max_wait_seconds": 6 * 60 * 60,
            "max_aicore_percent": 5,
            "max_hbm_percent": 10,
            "topology_aware": True,
            "lock_dir": "/tmp/vllm_ascend_parallel_bench_locks",
        },
        "execution": {
            "port_base": 18000,
            "max_retries": 2,
            "startup_timeout_seconds": 900,
            "execution_mode": "aclgraph",
            "distributed_executor_backend": "mp",
            "profile_policy": "representative_and_boundary",
            "allow_service_mutation": True,
        },
    }


def load_spec(path: str | Path | None = None) -> dict[str, Any]:
    spec = default_spec()
    if path is not None:
        user_spec = json.loads(Path(path).read_text(encoding="utf-8"))
        _deep_merge(spec, user_spec)
    validate_spec(spec)
    return spec


def _deep_merge(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def validate_spec(spec: Mapping[str, Any]) -> None:
    model_path = str(spec["model"]["path"])
    if not model_path.startswith("/") or "\x00" in model_path:
        raise ValueError("model.path must be an absolute container path")
    max_dies = int(spec["resource"]["max_dies"])
    if not 1 <= max_dies <= MAX_ASCEND_DIES:
        raise ValueError(f"resource.max_dies must be in [1, {MAX_ASCEND_DIES}]")
    if int(spec["resource"]["poll_seconds"]) < 1:
        raise ValueError("resource.poll_seconds must be positive")
    if int(spec["resource"]["max_wait_seconds"]) < 1:
        raise ValueError("resource.max_wait_seconds must be positive")
    for field in ("input_tokens", "output_tokens", "concurrency"):
        values = spec["workload"][field]
        if not values or any(int(value) < 1 for value in values):
            raise ValueError(f"workload.{field} must contain positive integers")


def expand_quick_matrix(spec: Mapping[str, Any]) -> list[ParallelCase]:
    cases = [
        ParallelCase("R0", 1, 1, False, profile=False),
        ParallelCase("P1", 2, 1, False, profile=True),
        ParallelCase("T1", 2, 2, False, profile=True),
        ParallelCase("E1", 2, 2, True, profile=True),
        ParallelCase("T2", 4, 2, False, profile=False),
        ParallelCase("E2", 4, 2, True, profile=False),
    ]
    if str(spec["model"].get("kind", "")).lower() != "moe":
        cases = [case for case in cases if not case.ep]
    return cases


def expand_boundary_matrix(spec: Mapping[str, Any]) -> list[ParallelCase]:
    max_dies = min(int(spec["resource"]["max_dies"]), MAX_ASCEND_DIES)
    is_moe = str(spec["model"].get("kind", "")).lower() == "moe"
    cases: list[ParallelCase] = []
    for pp in (2, 4, 8):
        for tp in (1, 2, 4):
            for ep in ((False, True) if is_moe else (False,)):
                case = ParallelCase(
                    case_id=f"P{pp}T{tp}{'E1' if ep else 'E0'}",
                    pp=pp,
                    tp=tp,
                    ep=ep,
                    profile=(pp, tp) in {(2, 1), (2, 2), (8, 2)},
                )
                if case.world_size <= max_dies:
                    cases.append(case)
    return cases


def expand_cases(spec: Mapping[str, Any]) -> list[ParallelCase]:
    preset = str(spec["matrix"].get("preset", "quick"))
    explicit = spec["matrix"].get("cases", [])
    if explicit:
        cases = [
            ParallelCase(
                case_id=str(item["case_id"]),
                pp=int(item.get("pp", 1)),
                tp=int(item.get("tp", 1)),
                ep=bool(item.get("ep", False)),
                cp=int(item.get("cp", 1)),
                profile=bool(item.get("profile", False)),
            )
            for item in explicit
        ]
    elif preset == "custom":
        raise ValueError("custom matrix preset requires at least one case")
    elif preset == "boundary":
        cases = expand_boundary_matrix(spec)
    else:
        cases = expand_quick_matrix(spec)
    max_dies = int(spec["resource"]["max_dies"])
    invalid = [case.case_id for case in cases if case.world_size > max_dies]
    if invalid:
        raise ValueError(f"cases exceed max_dies: {', '.join(invalid)}")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("case_id values must be unique")
    return cases


def render_parallel_flags(
    *,
    pp: int,
    tp: int,
    ep: bool,
    cp: int,
    capabilities: CapabilitySet,
) -> FlagDecision:
    if pp < 1 or tp < 1 or cp < 1:
        raise ValueError("pp, tp, and cp must be positive")
    flags = [
        "--pipeline-parallel-size",
        str(pp),
        "--tensor-parallel-size",
        str(tp),
    ]
    if ep:
        if not capabilities.expert_parallel_flag:
            return FlagDecision(
                (),
                "SKIPPED_UNSUPPORTED",
                "expert parallel is not supported by this vLLM CLI",
            )
        flags.append(capabilities.expert_parallel_flag)
    if cp > 1:
        if not capabilities.context_parallel_flag:
            return FlagDecision(
                (),
                "SKIPPED_UNSUPPORTED",
                "context parallel is not supported by this vLLM CLI",
            )
        flags.extend([capabilities.context_parallel_flag, str(cp)])
    return FlagDecision(tuple(flags))


def write_default_spec(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(default_spec(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
