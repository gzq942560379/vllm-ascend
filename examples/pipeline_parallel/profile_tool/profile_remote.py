#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run one vLLM-Ascend request profile from an Ascend Docker host.

This file is uploaded and invoked by ``profile_trace.ps1``.  It deliberately
uses only the Python standard library on the host.  NPU trace analysis runs
inside the vLLM-Ascend container where ``torch_npu`` is installed.
"""

import argparse
import base64
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional, Sequence


DEFAULT_KV_CACHE_BYTES = 4 * 1024**3
CONTAINER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
DEVICE_PATTERN = re.compile(r"^[0-9]+$")
EXECUTION_MODES = ("eager", "aclgraph")


@dataclass(frozen=True)
class ProfileConfig:
    container: str
    model: str
    device: str
    port: int = 8010
    prompt: str = "请用三句话说明流水线并行的工作原理。"
    max_tokens: int = 16
    num_requests: int = 1
    concurrency: int = 1
    execution_mode: str = "eager"
    expected_vllm_version: str = ""
    max_model_len: int = 4096
    kv_cache_memory_bytes: int = DEFAULT_KV_CACHE_BYTES
    startup_timeout_seconds: int = 300
    request_timeout_seconds: int = 600

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "ProfileConfig":
        config = cls(
            container=str(values.get("container", "")),
            model=str(values.get("model", "")),
            device=str(values.get("device", "")),
            port=int(values.get("port", 8010)),
            prompt=str(
                values.get(
                    "prompt",
                    "请用三句话说明流水线并行的工作原理。",
                )
            ),
            max_tokens=int(values.get("max_tokens", 16)),
            num_requests=int(values.get("num_requests", 1)),
            concurrency=int(values.get("concurrency", 1)),
            execution_mode=str(values.get("execution_mode", "eager")).lower(),
            expected_vllm_version=str(
                values.get("expected_vllm_version", "")
            ).strip(),
            max_model_len=int(values.get("max_model_len", 4096)),
            kv_cache_memory_bytes=int(
                values.get("kv_cache_memory_bytes", DEFAULT_KV_CACHE_BYTES)
            ),
            startup_timeout_seconds=int(
                values.get("startup_timeout_seconds", 300)
            ),
            request_timeout_seconds=int(
                values.get("request_timeout_seconds", 600)
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not CONTAINER_NAME_PATTERN.fullmatch(self.container):
            raise ValueError(
                "container must contain only letters, digits, dot, underscore, "
                "and hyphen"
            )
        if not self.model.startswith("/") or "\x00" in self.model:
            raise ValueError("model must be an absolute container path")
        if not DEVICE_PATTERN.fullmatch(self.device):
            raise ValueError("device must be one numeric NPU ID, for example 0")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be in the interval [1, 65535]")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.num_requests < 1:
            raise ValueError("num_requests must be >= 1")
        if not 1 <= self.concurrency <= self.num_requests:
            raise ValueError(
                "concurrency must be >= 1 and <= num_requests "
                f"({self.num_requests})"
            )
        if self.execution_mode not in EXECUTION_MODES:
            raise ValueError(
                f"execution_mode must be one of {EXECUTION_MODES}, "
                f"got {self.execution_mode!r}"
            )
        if self.max_model_len < 1:
            raise ValueError("max_model_len must be >= 1")
        if self.kv_cache_memory_bytes < 1:
            raise ValueError("kv_cache_memory_bytes must be >= 1")
        if self.startup_timeout_seconds < 10:
            raise ValueError("startup_timeout_seconds must be >= 10")
        if self.request_timeout_seconds < 10:
            raise ValueError("request_timeout_seconds must be >= 10")


def build_vllm_command(
    config: ProfileConfig,
    served_model_name: str,
    profile_dir: str,
) -> list[str]:
    profiler_config = {
        "profiler": "torch",
        "torch_profiler_dir": profile_dir,
        "torch_profiler_record_shapes": True,
        "torch_profiler_with_memory": True,
        "torch_profiler_with_stack": False,
    }
    command = [
        "vllm",
        "serve",
        config.model,
        "--served-model-name",
        served_model_name,
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
        "--distributed-executor-backend",
        "mp",
        "--max-model-len",
        str(config.max_model_len),
        "--max-num-seqs",
        str(config.concurrency),
        "--kv-cache-memory-bytes",
        str(config.kv_cache_memory_bytes),
        "--host",
        "0.0.0.0",
        "--port",
        str(config.port),
        "--profiler-config",
        json.dumps(profiler_config, separators=(",", ":")),
    ]
    if config.execution_mode == "eager":
        command.append("--enforce-eager")
    return command


def build_chat_request(
    config: ProfileConfig,
    served_model_name: str,
) -> dict:
    return {
        "model": served_model_name,
        "messages": [
            {
                "role": "user",
                "content": config.prompt,
            }
        ],
        "temperature": 0,
        "max_tokens": config.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def validate_vllm_version(actual: str, expected: str) -> None:
    if not expected:
        return
    suffix = actual[len(expected) :] if actual.startswith(expected) else ""
    if actual == expected or (suffix and suffix[0] in "+.-"):
        return
    raise RuntimeError(
        f"vLLM version mismatch: expected {expected!r}, found {actual!r}"
    )


def find_trace_view(profile_root: Path) -> Path:
    traces = sorted(profile_root.glob("**/ASCEND_PROFILER_OUTPUT/trace_view.json"))
    if len(traces) != 1:
        raise RuntimeError(
            f"Expected exactly one trace_view.json under {profile_root}, "
            f"found {len(traces)}"
        )
    return traces[0]


def _run(
    command: Sequence[str],
    *,
    input_text: Optional[str] = None,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        input=input_text,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=capture_output,
        check=False,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"Command failed ({result.returncode}): {shlex.join(command)}\n{detail}"
        )
    return result


def _docker_exec(
    container: str,
    command: Sequence[str],
    *,
    input_text: Optional[str] = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return _run(
        ["docker", "exec", "-i", container, *command],
        input_text=input_text,
        check=check,
    )


HTTP_CLIENT = r"""
import sys
import urllib.request

method, url = sys.argv[1:3]
body = sys.stdin.buffer.read()
data = body if body else (b"" if method == "POST" else None)
request = urllib.request.Request(
    url,
    data=data,
    method=method,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=10) as response:
    sys.stdout.buffer.write(response.read())
"""

BATCH_HTTP_CLIENT = r"""
import concurrent.futures
import json
import sys
import time
import urllib.request

config = json.load(sys.stdin)
url = config["url"]
payload = json.dumps(config["request"], ensure_ascii=False).encode("utf-8")
num_requests = config["num_requests"]
concurrency = config["concurrency"]
timeout = config["timeout_seconds"]

def send_one(index):
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return {
                "index": index,
                "http_status": response.status,
                "response": json.loads(body),
            }
    except Exception as error:
        return {"index": index, "error": repr(error)}

started = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
    results = list(executor.map(send_one, range(num_requests)))
elapsed = time.perf_counter() - started
successful = sum("error" not in item for item in results)
summary = {
    "num_requests": num_requests,
    "concurrency": concurrency,
    "successful": successful,
    "failed": num_requests - successful,
    "elapsed_seconds": elapsed,
    "responses": results,
}
print(json.dumps(summary, ensure_ascii=False))
if successful != num_requests:
    raise SystemExit(2)
"""

VERSION_INSPECTOR = r"""
import importlib.metadata
import json

packages = {
    "vllm": "vllm",
    "vllm_ascend": "vllm-ascend",
    "torch": "torch",
    "torch_npu": "torch-npu",
}
versions = {}
for label, distribution in packages.items():
    try:
        versions[label] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        versions[label] = "not-installed"
print(json.dumps(versions, sort_keys=True))
"""


def _container_http(
    config: ProfileConfig,
    method: str,
    path: str,
    payload: Optional[dict] = None,
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    body = "" if payload is None else json.dumps(payload, ensure_ascii=False)
    return _docker_exec(
        config.container,
        [
            "python3",
            "-c",
            HTTP_CLIENT,
            method,
            f"http://127.0.0.1:{config.port}{path}",
        ],
        input_text=body,
        check=check,
    )


def _container_chat_batch(
    config: ProfileConfig,
    request: dict,
) -> subprocess.CompletedProcess[str]:
    batch_config = {
        "url": (
            f"http://127.0.0.1:{config.port}/v1/chat/completions"
        ),
        "request": request,
        "num_requests": config.num_requests,
        "concurrency": config.concurrency,
        "timeout_seconds": config.request_timeout_seconds,
    }
    return _docker_exec(
        config.container,
        ["python3", "-c", BATCH_HTTP_CLIENT],
        input_text=json.dumps(batch_config, ensure_ascii=False),
    )


def _get_runtime_versions(container: str) -> dict:
    result = _docker_exec(
        container,
        ["python3", "-c", VERSION_INSPECTOR],
    )
    try:
        versions = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Could not parse runtime version information: "
            f"{result.stdout!r}"
        ) from error
    if not isinstance(versions, dict):
        raise RuntimeError("Runtime version information was not a JSON object")
    return versions


def _wait_for_health(config: ProfileConfig, log_path: str) -> None:
    deadline = time.monotonic() + config.startup_timeout_seconds
    while time.monotonic() < deadline:
        result = _container_http(config, "GET", "/health", check=False)
        if result.returncode == 0:
            return
        time.sleep(2)
    tail = _docker_exec(
        config.container,
        ["bash", "-lc", f"tail -n 80 {shlex.quote(log_path)}"],
        check=False,
    )
    raise RuntimeError(
        "vLLM did not become healthy before the startup timeout.\n"
        f"Last service log lines:\n{tail.stdout}{tail.stderr}"
    )


def _analyse_in_container(container: str, profile_dir: str) -> str:
    analyse_script = r"""
import sys
from pathlib import Path
from torch_npu.profiler.profiler import analyse

root = Path(sys.argv[1])
profiles = sorted(root.glob("*ascend_pt"))
if len(profiles) != 1:
    raise SystemExit(
        f"Expected exactly one raw ascend_pt directory under {root}, "
        f"found {len(profiles)}"
    )
analyse(str(profiles[0]))
traces = sorted(profiles[0].glob("ASCEND_PROFILER_OUTPUT/trace_view.json"))
if len(traces) != 1:
    raise SystemExit(f"Expected one analyzed trace, found {len(traces)}")
print(f"TRACE_IN_CONTAINER={traces[0]}")
"""
    result = _docker_exec(
        container,
        ["python3", "-c", analyse_script, profile_dir],
    )
    for line in result.stdout.splitlines():
        if line.startswith("TRACE_IN_CONTAINER="):
            return line.split("=", 1)[1]
    raise RuntimeError(
        "Profiler analysis completed without returning TRACE_IN_CONTAINER.\n"
        f"{result.stdout}{result.stderr}"
    )


def _stop_service(container: str, pid_path: str, served_model_name: str) -> None:
    stop_script = f"""
pid_path={shlex.quote(pid_path)}
expected={shlex.quote(served_model_name)}
if [ -s "$pid_path" ]; then
    pid=$(cat "$pid_path")
    case "$pid" in
        *[!0-9]*|'') exit 0 ;;
    esac
    if [ -r "/proc/$pid/cmdline" ]; then
        cmd=$(tr '\\0' ' ' < "/proc/$pid/cmdline")
        case "$cmd" in
            *"$expected"*) kill "$pid" 2>/dev/null || true ;;
        esac
    fi
fi
"""
    _docker_exec(container, ["bash", "-lc", stop_script], check=False)


def _decode_config(encoded: str) -> ProfileConfig:
    try:
        raw = base64.b64decode(encoded, validate=True).decode("utf-8")
        values = json.loads(raw)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid base64 JSON config: {error}") from error
    if not isinstance(values, dict):
        raise ValueError("Decoded config must be a JSON object")
    return ProfileConfig.from_mapping(values)


def run_profile(config: ProfileConfig) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    served_model_name = f"qwen-profile-{run_id}"
    container_root = f"/workspace/profile_tool_runs/{run_id}"
    profile_dir = f"{container_root}/raw"
    log_path = f"{container_root}/service.log"
    pid_path = f"{container_root}/service.pid"
    command = build_vllm_command(config, served_model_name, profile_dir)
    export_dir = Path.home() / "vllm_profile_exports" / run_id
    export_dir.mkdir(parents=True, exist_ok=False)

    _run(["docker", "inspect", "--type", "container", config.container])
    _docker_exec(
        config.container,
        ["test", "-s", f"{config.model}/config.json"],
    )
    runtime_versions = _get_runtime_versions(config.container)
    actual_vllm_version = str(runtime_versions.get("vllm", "not-installed"))
    validate_vllm_version(
        actual_vllm_version,
        config.expected_vllm_version,
    )
    print(
        "Runtime versions: "
        + json.dumps(runtime_versions, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    occupied = _container_http(config, "GET", "/health", check=False)
    if occupied.returncode == 0:
        raise RuntimeError(
            f"Port {config.port} already has a healthy service inside "
            f"{config.container}; choose another -Port value"
        )
    _docker_exec(config.container, ["mkdir", "-p", profile_dir])

    launch_script = (
        f"echo $$ > {shlex.quote(pid_path)}; "
        f"exec {shlex.join(command)} > {shlex.quote(log_path)} 2>&1"
    )
    _run(
        [
            "docker",
            "exec",
            "-d",
            "-e",
            f"ASCEND_RT_VISIBLE_DEVICES={config.device}",
            "-e",
            "HF_HUB_OFFLINE=1",
            "-e",
            "TRANSFORMERS_OFFLINE=1",
            config.container,
            "bash",
            "-lc",
            launch_script,
        ]
    )

    profile_started = False
    response_text = ""
    try:
        print("Waiting for vLLM startup...", flush=True)
        _wait_for_health(config, log_path)
        request = build_chat_request(config, served_model_name)

        print("Sending warm-up request...", flush=True)
        _container_http(config, "POST", "/v1/chat/completions", request)

        print("Starting profiler and sending the measured request...", flush=True)
        _container_http(config, "POST", "/start_profile")
        profile_started = True
        time.sleep(0.5)
        print(
            f"Sending {config.num_requests} measured request(s) with "
            f"concurrency={config.concurrency}...",
            flush=True,
        )
        response = _container_chat_batch(config, request)
        response_text = response.stdout
        _container_http(config, "POST", "/stop_profile")
        profile_started = False

        raw_deadline = time.monotonic() + 60
        while time.monotonic() < raw_deadline:
            raw_check = _docker_exec(
                config.container,
                [
                    "bash",
                    "-lc",
                    f"find {shlex.quote(profile_dir)} -maxdepth 1 "
                    "-type d -name '*ascend_pt' -print -quit",
                ],
                check=False,
            )
            if raw_check.stdout.strip():
                break
            time.sleep(1)
        else:
            raise RuntimeError("Profiler did not create an ascend_pt directory")

        print("Analysing the raw Ascend profile...", flush=True)
        trace_in_container = _analyse_in_container(
            config.container,
            profile_dir,
        )
        trace_export = export_dir / "trace_view.json"
        _run(
            [
                "docker",
                "cp",
                f"{config.container}:{trace_in_container}",
                str(trace_export),
            ]
        )
        _run(
            [
                "docker",
                "cp",
                f"{config.container}:{log_path}",
                str(export_dir / "service.log"),
            ]
        )
        (export_dir / "response.json").write_text(
            response_text,
            encoding="utf-8",
        )
        manifest = {
            "run_id": run_id,
            "container": config.container,
            "model": config.model,
            "device": config.device,
            "port": config.port,
            "prompt": config.prompt,
            "max_tokens": config.max_tokens,
            "num_requests": config.num_requests,
            "concurrency": config.concurrency,
            "execution_mode": config.execution_mode,
            "expected_vllm_version": config.expected_vllm_version,
            "runtime_versions": runtime_versions,
            "trace_in_container": trace_in_container,
            "trace_export": str(trace_export),
            "vllm_command": command,
        }
        (export_dir / "run.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"TRACE_EXPORT={trace_export}", flush=True)
        print(f"RUN_EXPORT_DIR={export_dir}", flush=True)
        return trace_export
    finally:
        if profile_started:
            _container_http(config, "POST", "/stop_profile", check=False)
        _stop_service(config.container, pid_path, served_model_name)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-b64", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        config = _decode_config(args.config_b64)
        run_profile(config)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
