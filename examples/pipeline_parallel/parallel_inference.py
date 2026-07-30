#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and launch reproducible PP/TP/EP and weight-offload experiments."""

import argparse
import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

MODEL_KINDS = ("auto", "dense", "moe")
OFFLOAD_BACKENDS = ("none", "prefetch", "uva")
EXPERT_COUNT_KEYS = (
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
)


@dataclass(frozen=True)
class ServePlan:
    model: str
    devices: Tuple[str, ...]
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    enable_expert_parallel: bool = False
    model_kind: str = "auto"
    offload_backend: str = "none"
    offload_group_size: int = 0
    offload_num_in_group: int = 1
    offload_prefetch_step: int = 1
    offload_params: Tuple[str, ...] = ()
    cpu_offload_gb: float = 0
    enforce_eager: bool = False
    served_model_name: str = "qwen"
    max_model_len: int = 4096
    max_num_seqs: int = 16
    gpu_memory_utilization: float = 0.8
    kv_cache_memory_bytes: int = 0
    host: str = "0.0.0.0"
    port: int = 8000
    distributed_executor_backend: str = "mp"
    offline: bool = True
    extra_args: Tuple[str, ...] = ()


def _load_model_config(model: str) -> Optional[dict]:
    model_path = Path(model)
    config_path = model_path / "config.json" if model_path.is_dir() else None
    if config_path is None or not config_path.is_file():
        return None
    with config_path.open(encoding="utf-8") as config_file:
        return json.load(config_file)


def detect_model_kind(model: str) -> str:
    """Classify a local checkpoint as dense, MoE, or unknown."""
    config = _load_model_config(model)
    if config is None:
        return "unknown"

    for key in EXPERT_COUNT_KEYS:
        expert_count = config.get(key)
        if isinstance(expert_count, int) and expert_count > 0:
            return "moe"

    architectures = config.get("architectures") or []
    if any("moe" in str(architecture).lower() for architecture in architectures):
        return "moe"
    return "dense"


def _resolved_model_kind(plan: ServePlan) -> str:
    if plan.model_kind not in MODEL_KINDS:
        raise ValueError(f"model_kind must be one of {MODEL_KINDS}, got {plan.model_kind!r}")
    if plan.model_kind != "auto":
        return plan.model_kind
    return detect_model_kind(plan.model)


def validate_plan(plan: ServePlan) -> None:
    if not plan.model:
        raise ValueError("model path or model ID must not be empty")
    if plan.tensor_parallel_size < 1 or plan.pipeline_parallel_size < 1:
        raise ValueError("tensor_parallel_size and pipeline_parallel_size must both be >= 1")

    devices = tuple(device.strip() for device in plan.devices if device.strip())
    if len(devices) != len(set(devices)):
        raise ValueError(f"devices must be unique, got {devices}")
    expected_world_size = plan.pipeline_parallel_size * plan.tensor_parallel_size
    if len(devices) != expected_world_size:
        raise ValueError(
            f"Visible device count must match PP({plan.pipeline_parallel_size}) × "
            f"TP({plan.tensor_parallel_size}) = {expected_world_size}, got {len(devices)}"
        )

    if not 0 < plan.gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in the interval (0, 1]")
    if plan.kv_cache_memory_bytes < 0:
        raise ValueError("kv_cache_memory_bytes must be >= 0")
    if plan.max_model_len < 1 or plan.max_num_seqs < 1:
        raise ValueError("max_model_len and max_num_seqs must both be >= 1")
    if not 1 <= plan.port <= 65535:
        raise ValueError("port must be in the interval [1, 65535]")

    model_kind = _resolved_model_kind(plan)
    if plan.enable_expert_parallel:
        if model_kind == "dense":
            raise ValueError(
                "Expert parallelism cannot be enabled for a Dense checkpoint. "
                "Use Qwen3-8B for PP+TP, then switch to a MoE checkpoint for PP+EP."
            )
        if model_kind == "unknown":
            raise ValueError(
                "Cannot determine whether this remote checkpoint is MoE. "
                "Use a local config.json or pass --model-kind moe explicitly."
            )

    if plan.offload_backend not in OFFLOAD_BACKENDS:
        raise ValueError(f"offload_backend must be one of {OFFLOAD_BACKENDS}, got {plan.offload_backend!r}")
    if plan.offload_backend == "prefetch":
        if plan.offload_group_size < 1:
            raise ValueError("offload_group_size must be >= 1 for prefetch offloading")
        if not 1 <= plan.offload_num_in_group <= plan.offload_group_size:
            raise ValueError(
                "offload_num_in_group must be >= 1 and <= offload_group_size "
                f"({plan.offload_group_size}), got {plan.offload_num_in_group}"
            )
        if plan.offload_prefetch_step < 1:
            raise ValueError("offload_prefetch_step must be >= 1 for prefetch offloading")
        if plan.cpu_offload_gb:
            raise ValueError("cpu_offload_gb belongs to the UVA backend and cannot be combined with prefetch")
    elif plan.offload_backend == "uva":
        if plan.cpu_offload_gb <= 0:
            raise ValueError("cpu_offload_gb must be > 0 for the UVA backend")
        if not plan.enforce_eager:
            raise ValueError(
                "Ascend has no UVA zero-copy path; its functional-call fallback requires --enforce-eager"
            )
        if plan.offload_group_size:
            raise ValueError("offload_group_size belongs to the prefetch backend and cannot be combined with UVA")
    elif plan.cpu_offload_gb or plan.offload_group_size:
        raise ValueError("Select --offload-backend prefetch or uva when configuring weight offloading")


def build_vllm_command(plan: ServePlan) -> list:
    validate_plan(plan)
    command = [
        "vllm",
        "serve",
        plan.model,
        "--served-model-name",
        plan.served_model_name,
        "--tensor-parallel-size",
        str(plan.tensor_parallel_size),
        "--pipeline-parallel-size",
        str(plan.pipeline_parallel_size),
        "--distributed-executor-backend",
        plan.distributed_executor_backend,
        "--max-model-len",
        str(plan.max_model_len),
        "--max-num-seqs",
        str(plan.max_num_seqs),
    ]
    if plan.kv_cache_memory_bytes:
        command.extend(["--kv-cache-memory-bytes", str(plan.kv_cache_memory_bytes)])
    else:
        command.extend(["--gpu-memory-utilization", str(plan.gpu_memory_utilization)])
    command.extend(["--host", plan.host, "--port", str(plan.port)])

    if plan.enable_expert_parallel:
        command.append("--enable-expert-parallel")
    if plan.offload_backend == "prefetch":
        command.extend(
            [
                "--offload-backend",
                "prefetch",
                "--offload-group-size",
                str(plan.offload_group_size),
                "--offload-num-in-group",
                str(plan.offload_num_in_group),
                "--offload-prefetch-step",
                str(plan.offload_prefetch_step),
            ]
        )
        if plan.offload_params:
            command.append("--offload-params")
            command.extend(plan.offload_params)
    elif plan.offload_backend == "uva":
        command.extend(
            [
                "--offload-backend",
                "uva",
                "--cpu-offload-gb",
                str(plan.cpu_offload_gb),
            ]
        )

    if plan.enforce_eager:
        command.append("--enforce-eager")
    command.extend(plan.extra_args)
    return command


def build_environment(plan: ServePlan, current_environment: Mapping[str, str]) -> dict:
    validate_plan(plan)
    environment = dict(current_environment)
    environment["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(plan.devices)
    if plan.offline:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    if plan.offload_backend == "uva":
        # Ascend does not provide CUDA UVA. Force vLLM's functional-call fallback.
        environment["VLLM_WEIGHT_OFFLOADING_DISABLE_UVA"] = "1"
    return environment


def _comma_separated(value: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch validated vLLM-Ascend PP/TP/EP and weight-offload experiments."
    )
    parser.add_argument("--model", required=True, help="Local checkpoint directory or model ID.")
    parser.add_argument("--devices", required=True, type=_comma_separated, help="Visible NPU IDs, for example 0,1,2,3.")
    parser.add_argument("--tp", type=int, default=1, dest="tensor_parallel_size")
    parser.add_argument("--pp", type=int, default=1, dest="pipeline_parallel_size")
    parser.add_argument("--enable-ep", action="store_true", dest="enable_expert_parallel")
    parser.add_argument("--model-kind", choices=MODEL_KINDS, default="auto")
    parser.add_argument("--offload-backend", choices=OFFLOAD_BACKENDS, default="none")
    parser.add_argument("--offload-group-size", type=int, default=0)
    parser.add_argument("--offload-num-in-group", type=int, default=1)
    parser.add_argument("--offload-prefetch-step", type=int, default=1)
    parser.add_argument(
        "--offload-params",
        type=_comma_separated,
        default=(),
        help="Comma-separated parameter name segments, for example gate_up_proj,down_proj.",
    )
    parser.add_argument("--cpu-offload-gb", type=float, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--served-model-name", default="qwen")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--kv-cache-memory-bytes",
        type=int,
        default=0,
        help="Fix KV cache bytes per NPU and omit --gpu-memory-utilization; 0 keeps automatic sizing.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--distributed-executor-backend", default="mp")
    parser.add_argument("--online", action="store_false", dest="offline", help="Allow remote model/tokenizer access.")
    parser.add_argument("--dry-run", action="store_true", help="Print the validated launch plan without executing it.")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra vllm arguments after '--', kept as separate argv tokens.",
    )
    return parser


def _plan_from_args(args: argparse.Namespace) -> ServePlan:
    extra_args = tuple(args.extra_args)
    if extra_args[:1] == ("--",):
        extra_args = extra_args[1:]
    return ServePlan(
        model=args.model,
        devices=args.devices,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
        model_kind=args.model_kind,
        offload_backend=args.offload_backend,
        offload_group_size=args.offload_group_size,
        offload_num_in_group=args.offload_num_in_group,
        offload_prefetch_step=args.offload_prefetch_step,
        offload_params=args.offload_params,
        cpu_offload_gb=args.cpu_offload_gb,
        enforce_eager=args.enforce_eager,
        served_model_name=args.served_model_name,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        host=args.host,
        port=args.port,
        distributed_executor_backend=args.distributed_executor_backend,
        offline=args.offline,
        extra_args=extra_args,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    plan = _plan_from_args(args)
    try:
        command = build_vllm_command(plan)
        environment = build_environment(plan, os.environ)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    visible_environment = {
        key: environment[key]
        for key in (
            "ASCEND_RT_VISIBLE_DEVICES",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "VLLM_WEIGHT_OFFLOADING_DISABLE_UVA",
        )
        if key in environment
    }
    print(json.dumps({"environment": visible_environment, "command": command}, ensure_ascii=False, indent=2))
    print(f"command: {shlex.join(command)}")
    if args.dry_run:
        return 0

    os.execvpe(command[0], command, environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
