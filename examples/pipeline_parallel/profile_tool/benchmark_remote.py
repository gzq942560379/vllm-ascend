#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Persistent remote controller for a recoverable parallel benchmark run."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiment_schema import (
    CapabilitySet,
    ParallelCase,
    expand_cases,
    load_spec,
    render_parallel_flags,
)
from resource_scheduler import discover_idle_devices, wait_for_lease
from workload_streaming import (
    build_exact_prompt,
    percentile,
    run_workload,
    tokenize_count,
)
from report_generator import generate_report
from metrics_analyzer import analyze_bubbles, analyze_kernel_csv


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CaseStatus(str, Enum):
    QUEUED = "QUEUED"
    WAIT_NPU = "WAIT_NPU"
    LEASED = "LEASED"
    PREPARE = "PREPARE"
    WARMUP = "WARMUP"
    BENCHMARK = "BENCHMARK"
    PROFILE = "PROFILE"
    ANALYZE = "ANALYZE"
    RETRYABLE = "RETRYABLE"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    SKIPPED_CAPACITY = "SKIPPED_CAPACITY"
    SKIPPED_UNSUPPORTED = "SKIPPED_UNSUPPORTED"


TERMINAL_STATUSES = {
    CaseStatus.COMPLETE,
    CaseStatus.FAILED,
    CaseStatus.SKIPPED_CAPACITY,
    CaseStatus.SKIPPED_UNSUPPORTED,
    CaseStatus.CANCELLED,
}


class RunCancelled(RuntimeError):
    pass


@dataclass
class CaseState:
    case_id: str
    status: CaseStatus = CaseStatus.QUEUED
    attempts: int = 0
    reason: str = ""
    devices: list[int] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now)


@dataclass
class ExperimentState:
    run_id: str
    status: str
    cases: dict[str, CaseState]
    created_at: str
    updated_at: str
    heartbeat: str = ""

    @classmethod
    def new(cls, run_id: str, case_ids: Sequence[str]) -> "ExperimentState":
        now = utc_now()
        return cls(
            run_id=run_id,
            status="QUEUED",
            cases={case_id: CaseState(case_id) for case_id in case_ids},
            created_at=now,
            updated_at=now,
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = utc_now()
        payload = asdict(self)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentState":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["cases"] = {
            key: CaseState(
                case_id=value["case_id"],
                status=CaseStatus(value["status"]),
                attempts=int(value.get("attempts", 0)),
                reason=str(value.get("reason", "")),
                devices=[int(item) for item in value.get("devices", [])],
                updated_at=str(value.get("updated_at", utc_now())),
            )
            for key, value in payload["cases"].items()
        }
        return cls(**payload)


def runnable_case_ids(state: ExperimentState) -> list[str]:
    return [
        case_id
        for case_id, case in state.cases.items()
        if case.status not in TERMINAL_STATUSES
    ]


def _docker_exec(container: str, command: Sequence[str], **kwargs: Any):
    return subprocess.run(["docker", "exec", container, *command], **kwargs)


def probe_capabilities(container: str, output_dir: Path) -> CapabilitySet:
    help_result = _docker_exec(
        container,
        ["vllm", "serve", "--help=all"],
        capture_output=True,
        text=True,
    )
    help_text = (help_result.stdout or "") + (help_result.stderr or "")
    capabilities = CapabilitySet.from_help_text(help_text)
    version_script = (
        "import importlib.metadata as m,json;"
        "names=['vllm','vllm-ascend','torch','torch-npu'];"
        "out={};"
        "\nfor n in names:\n"
        " try: out[n]=m.version(n)\n"
        " except m.PackageNotFoundError: out[n]=''\n"
        "print(json.dumps(out))"
    )
    versions = _docker_exec(
        container,
        ["python3", "-c", version_script],
        capture_output=True,
        text=True,
    )
    try:
        version_values = json.loads(versions.stdout.strip() or "{}")
    except json.JSONDecodeError:
        version_values = {}
    environment = {
        "captured_at": utc_now(),
        "container": container,
        "capabilities": capabilities.as_dict(),
        "versions": version_values,
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return capabilities


def aggregate_request_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    successful = [row for row in rows if row.get("ok")]
    result: dict[str, Any] = {
        "requests": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
    }
    for source, target in (
        ("ttft_ms", "ttft"),
        ("tpot_ms", "tpot"),
        ("e2e_ms", "e2e"),
    ):
        values = [
            float(row[source])
            for row in successful
            if row.get(source) is not None
        ]
        for quantile in (50, 95, 99):
            result[f"{target}_p{quantile}_ms"] = percentile(values, quantile)
    total_tokens = sum(int(row.get("output_tokens", 0)) for row in successful)
    starts = [
        float(row["started_at_monotonic"])
        for row in successful
        if row.get("started_at_monotonic") is not None
    ]
    ends = [
        float(row["completed_at_monotonic"])
        for row in successful
        if row.get("completed_at_monotonic") is not None
    ]
    total_seconds = max(ends) - min(starts) if starts and ends else 0.0
    result["output_throughput"] = (
        total_tokens / total_seconds if total_seconds else 0.0
    )
    result["request_throughput"] = (
        len(successful) / total_seconds if total_seconds else 0.0
    )
    return result


class RemoteController:

    def __init__(self, run_dir: Path, spec_path: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.spec = load_spec(spec_path)
        self.cases = expand_cases(self.spec)
        self.state_path = run_dir / "state.json"
        self.state = (
            ExperimentState.load(self.state_path)
            if self.state_path.exists()
            else ExperimentState.new(
                run_dir.name, [case.case_id for case in self.cases]
            )
        )
        self.selector = Path(__file__).resolve().parent.parent / "find_idle_npu.sh"
        self.capabilities: CapabilitySet | None = None

    def update(
        self,
        case: ParallelCase | None,
        status: CaseStatus | str,
        reason: str = "",
    ) -> None:
        value = status.value if isinstance(status, CaseStatus) else status
        self.state.status = value
        self.state.heartbeat = f"{utc_now()} {value} {reason}".strip()
        if case is not None:
            case_state = self.state.cases[case.case_id]
            case_state.status = CaseStatus(value)
            case_state.reason = reason
            case_state.updated_at = utc_now()
        self.state.save(self.state_path)

    def run(self) -> int:
        container = str(self.spec["container"]["name"])
        try:
            self.capabilities = probe_capabilities(container, self.run_dir)
            self._validate_environment(container)
        except Exception as exc:
            self.state.status = "FAILED"
            self.state.heartbeat = (
                f"{utc_now()} PREFLIGHT_FAILED {type(exc).__name__}: {exc}"
            )
            self.state.save(self.state_path)
            self._generate_report()
            return 1
        by_id = {case.case_id: case for case in self.cases}
        pending = runnable_case_ids(self.state)
        safety_rounds = 0
        while pending:
            safety_rounds += 1
            if safety_rounds > (int(self.spec["execution"]["max_retries"]) + 2):
                break
            for case_id in pending:
                if self._cancel_requested():
                    self.update(by_id[case_id], CaseStatus.CANCELLED)
                    continue
                self.run_case(by_id[case_id])
            pending = runnable_case_ids(self.state)
        failures = [
            case
            for case in self.state.cases.values()
            if case.status == CaseStatus.FAILED
        ]
        cancelled = any(
            case.status == CaseStatus.CANCELLED
            for case in self.state.cases.values()
        )
        unfinished = runnable_case_ids(self.state)
        if failures or unfinished:
            self.state.status = "FAILED"
        elif cancelled:
            self.state.status = "CANCELLED"
        else:
            self.state.status = "COMPLETE"
        self.state.save(self.state_path)
        self._generate_report()
        return 1 if failures or unfinished else 0

    def _validate_environment(self, container: str) -> None:
        model_path = str(self.spec["model"]["container_path"])
        check = _docker_exec(
            container,
            ["test", "-s", f"{model_path}/config.json"],
            capture_output=True,
        )
        if check.returncode:
            raise RuntimeError(
                f"model config is not visible in container: {model_path}"
            )
        inspect_script = (
            "import json,sys;"
            "c=json.load(open(sys.argv[1]+'/config.json'));"
            "keys=['architectures','model_type','torch_dtype',"
            "'num_hidden_layers','num_experts','num_experts_per_tok',"
            "'moe_intermediate_size'];"
            "print(json.dumps({k:c.get(k) for k in keys}))"
        )
        inspected = _docker_exec(
            container,
            ["python3", "-c", inspect_script, model_path],
            capture_output=True,
            text=True,
        )
        if inspected.returncode:
            raise RuntimeError(
                "failed to inspect model config: " + inspected.stderr[-1000:]
            )
        model_metadata = json.loads(inspected.stdout.strip())
        expert_count = int(model_metadata.get("num_experts") or 0)
        architecture_text = " ".join(
            str(value) for value in model_metadata.get("architectures") or []
        )
        actual_kind = (
            "moe"
            if expert_count > 0 or "moe" in architecture_text.lower()
            else "dense"
        )
        configured_kind = str(self.spec["model"].get("kind", "auto")).lower()
        if configured_kind == "auto":
            self.spec["model"]["kind"] = actual_kind
            self.cases = expand_cases(self.spec)
            for case in self.cases:
                self.state.cases.setdefault(
                    case.case_id, CaseState(case.case_id)
                )
        elif configured_kind != actual_kind:
            raise RuntimeError(
                f"model kind mismatch: spec={configured_kind}, "
                f"config.json={actual_kind}"
            )
        environment = json.loads(
            (self.run_dir / "environment.json").read_text(encoding="utf-8")
        )
        versions = environment.get("versions", {})
        environment["model"] = {
            "host_path": str(self.spec["model"]["path"]),
            "path": model_path,
            "kind": actual_kind,
            **model_metadata,
        }
        (self.run_dir / "environment.json").write_text(
            json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        expected = {
            "vllm": self.spec["container"].get("expected_vllm_version", ""),
            "vllm-ascend": self.spec["container"].get(
                "expected_vllm_ascend_version", ""
            ),
        }
        for package, wanted in expected.items():
            actual = str(versions.get(package, ""))
            if wanted and not actual.startswith(str(wanted)):
                raise RuntimeError(
                    f"{package} version mismatch: expected {wanted}, got {actual}"
                )

    def run_case(self, case: ParallelCase) -> None:
        assert self.capabilities is not None
        decision = render_parallel_flags(
            pp=case.pp,
            tp=case.tp,
            ep=case.ep,
            cp=case.cp,
            capabilities=self.capabilities,
        )
        if decision.status:
            self.update(case, decision.status, decision.reason)
            return
        resource = self.spec["resource"]
        case_state = self.state.cases[case.case_id]
        self.update(case, CaseStatus.WAIT_NPU)

        def heartbeat(message: str) -> None:
            if self._cancel_requested():
                raise RunCancelled("cancellation requested while waiting for NPU")
            self.update(case, CaseStatus.WAIT_NPU, message)

        try:
            lease = wait_for_lease(
                count=case.world_size,
                discover=lambda: discover_idle_devices(
                    self.selector,
                    max_aicore=int(resource["max_aicore_percent"]),
                    max_hbm_pct=int(resource["max_hbm_percent"]),
                ),
                lock_dir=resource["lock_dir"],
                poll_seconds=int(resource["poll_seconds"]),
                max_wait_seconds=int(resource["max_wait_seconds"]),
                heartbeat=heartbeat,
            )
        except RunCancelled as exc:
            self.update(case, CaseStatus.CANCELLED, str(exc))
            return
        if lease is None:
            self.update(case, CaseStatus.FAILED, "timed out waiting for idle NPU")
            return
        try:
            case_state.devices = [
                device.logic_id for device in lease.devices
            ]
            case_state.attempts += 1
            self.update(case, CaseStatus.LEASED)
            self._execute_case(case, decision.flags)
        except Exception as exc:
            service_log = (
                self.run_dir / "cases" / case.case_id / "service.log"
            )
            diagnostic = f"{type(exc).__name__}: {exc}"
            if service_log.is_file():
                diagnostic += "\n" + service_log.read_text(
                    encoding="utf-8", errors="replace"
                )[-4000:]
            if re.search(
                r"out of memory|\boom\b|memory allocation|alloc.*failed",
                diagnostic,
                re.I,
            ):
                self.update(
                    case,
                    CaseStatus.SKIPPED_CAPACITY,
                    diagnostic[-1000:],
                )
                return
            retries = int(self.spec["execution"]["max_retries"])
            if case_state.attempts <= retries:
                self.update(
                    case,
                    CaseStatus.RETRYABLE,
                    f"{type(exc).__name__}: {exc}",
                )
            else:
                self.update(
                    case, CaseStatus.FAILED, f"{type(exc).__name__}: {exc}"
                )
        finally:
            lease.release()

    def _execute_case(
        self, case: ParallelCase, parallel_flags: Sequence[str]
    ) -> None:
        case_dir = self.run_dir / "cases" / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        # The durable controller owns orchestration and always records command
        # evidence before it mutates the run-specific service state.
        command_evidence = {
            "case": case.as_dict(),
            "parallel_flags": list(parallel_flags),
            "devices": self.state.cases[case.case_id].devices,
        }
        (case_dir / "command.json").write_text(
            json.dumps(command_evidence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.update(case, CaseStatus.PREPARE)
        # A fully automatic container service lifecycle is intentionally
        # guarded by this flag. It prevents accidental interference with an
        # existing shared service during initial deployment.
        if not self.spec["execution"].get("allow_service_mutation", False):
            self.update(
                case,
                CaseStatus.SKIPPED_UNSUPPORTED,
                "set execution.allow_service_mutation=true after smoke validation",
            )
            return
        container = str(self.spec["container"]["name"])
        model = self.spec["model"]
        workload = self.spec["workload"]
        execution = self.spec["execution"]
        port = self._find_free_port(
            int(execution["port_base"]), f"{self.run_dir.name}-{case.case_id}"
        )
        served_name = f"{model['served_name']}-{case.case_id.lower()}"
        devices = self.state.cases[case.case_id].devices
        container_run_dir = (
            f"/workspace/parallel_bench_runs/{self.run_dir.name}"
        )
        container_case_dir = f"{container_run_dir}/cases/{case.case_id}"
        service_log = f"{container_case_dir}/service.log"
        pid_file = f"{container_case_dir}/service.pid"
        command = [
            "vllm",
            "serve",
            str(model["container_path"]),
            "--served-model-name",
            served_name,
            "--distributed-executor-backend",
            str(execution["distributed_executor_backend"]),
            "--max-model-len",
            str(model["max_model_len"]),
            "--max-num-seqs",
            str(max(int(value) for value in workload["concurrency"])),
            "--gpu-memory-utilization",
            "0.8",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            *parallel_flags,
        ]
        if str(execution["execution_mode"]) == "eager":
            command.append("--enforce-eager")
        profile_root = f"{container_run_dir}/profiles/{case.case_id}"
        if case.profile and self.capabilities and self.capabilities.profiler_config:
            profile_config = {
                "profiler": "torch",
                "torch_profiler_dir": profile_root,
                "torch_profiler_record_shapes": True,
                "torch_profiler_with_memory": True,
                "torch_profiler_with_stack": False,
            }
            command.extend(
                [
                    "--profiler-config",
                    json.dumps(profile_config, separators=(",", ":")),
                ]
            )
        command_evidence["vllm_command"] = command
        (case_dir / "command.json").write_text(
            json.dumps(command_evidence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shell_command = (
            f"mkdir -p {shlex.quote(container_case_dir)} "
            f"{shlex.quote(profile_root)}; "
            f"echo $$ > {shlex.quote(pid_file)}; "
            f"exec {shlex.join(command)} > {shlex.quote(service_log)} 2>&1"
        )
        launch = subprocess.run(
            [
                "docker",
                "exec",
                "-d",
                "-e",
                f"ASCEND_RT_VISIBLE_DEVICES={','.join(map(str, devices))}",
                "-e",
                "HF_HUB_OFFLINE=1",
                "-e",
                "TRANSFORMERS_OFFLINE=1",
                container,
                "bash",
                "-lc",
                shell_command,
            ],
            capture_output=True,
            text=True,
        )
        if launch.returncode:
            raise RuntimeError(launch.stderr.strip() or "docker exec failed")
        try:
            self._wait_for_health(port, int(execution["startup_timeout_seconds"]))
            base_url = f"http://127.0.0.1:{port}"
            prompt_cache = {
                int(target): build_exact_prompt(
                    int(target),
                    count_tokens=lambda prompt: tokenize_count(
                        url=base_url,
                        model=served_name,
                        prompt=prompt,
                        timeout_seconds=30,
                    ),
                )
                for target in workload["input_tokens"]
            }
            self.update(case, CaseStatus.WARMUP)
            warmup_rows = run_workload(
                url=base_url,
                model=served_name,
                input_tokens=int(workload["input_tokens"][0]),
                output_tokens=min(16, int(workload["output_tokens"][0])),
                num_requests=int(workload["warmup_requests"]),
                concurrency=min(
                    int(workload["warmup_requests"]),
                    int(workload["concurrency"][0]),
                ),
                timeout_seconds=int(workload["request_timeout_seconds"]),
                prompt=prompt_cache[int(workload["input_tokens"][0])],
            )
            if any(not row["ok"] for row in warmup_rows):
                raise RuntimeError("one or more warmup requests failed")
            self.update(case, CaseStatus.BENCHMARK)
            request_output = case_dir / "request_metrics.jsonl"
            summaries: list[dict[str, Any]] = []
            with request_output.open("a", encoding="utf-8") as output:
                for input_tokens in workload["input_tokens"]:
                    for output_tokens in workload["output_tokens"]:
                        for concurrency in workload["concurrency"]:
                            for repetition in range(int(workload["repetitions"])):
                                rows = run_workload(
                                    url=base_url,
                                    model=served_name,
                                    input_tokens=int(input_tokens),
                                    output_tokens=int(output_tokens),
                                    num_requests=int(
                                        workload["requests_per_point"]
                                    ),
                                    concurrency=int(concurrency),
                                    timeout_seconds=int(
                                        workload["request_timeout_seconds"]
                                    ),
                                    prompt=prompt_cache[int(input_tokens)],
                                )
                                for row in rows:
                                    enriched = {
                                        "case_id": case.case_id,
                                        "input_tokens_target": input_tokens,
                                        "output_tokens_target": output_tokens,
                                        "concurrency": concurrency,
                                        "repetition": repetition,
                                        **row,
                                    }
                                    output.write(
                                        json.dumps(enriched, ensure_ascii=False)
                                        + "\n"
                                    )
                                summary = aggregate_request_metrics(rows)
                                summary.update(
                                    {
                                        "case_id": case.case_id,
                                        "pp": case.pp,
                                        "tp": case.tp,
                                        "ep": case.ep,
                                        "cp": case.cp,
                                        "input_tokens": input_tokens,
                                        "output_tokens": output_tokens,
                                        "concurrency": concurrency,
                                        "repetition": repetition,
                                    }
                                )
                                summaries.append(summary)
                                self.state.heartbeat = (
                                    f"{utc_now()} BENCHMARK {case.case_id} "
                                    f"in={input_tokens} out={output_tokens} "
                                    f"c={concurrency} rep={repetition}"
                                )
                                self.state.save(self.state_path)
                                if self._cancel_requested():
                                    self.update(case, CaseStatus.CANCELLED)
                                    return
            (case_dir / "summary.json").write_text(
                json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if case.profile and self.capabilities and self.capabilities.profiler_config:
                self._profile_case(
                    case, port, served_name, container, profile_root
                )
            self.update(case, CaseStatus.COMPLETE)
        finally:
            self._stop_owned_service(container, pid_file)

    @staticmethod
    def _wait_for_health(port: int, timeout_seconds: int) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_error = ""
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=3
                ) as response:
                    if response.status == 200:
                        return
            except (OSError, urllib.error.URLError) as exc:
                last_error = str(exc)
            time.sleep(5)
        raise TimeoutError(f"vLLM health timeout on port {port}: {last_error}")

    @staticmethod
    def _find_free_port(base: int, key: str) -> int:
        first = base + zlib.crc32(key.encode("utf-8")) % 20000
        for offset in range(100):
            port = first + offset
            if port > 65535:
                port = base + offset
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                try:
                    probe.bind(("127.0.0.1", port))
                except OSError:
                    continue
                return port
        raise RuntimeError("no free benchmark port found")

    @staticmethod
    def _post(url: str) -> None:
        request = urllib.request.Request(url, data=b"", method="POST")
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status >= 300:
                raise RuntimeError(f"POST {url} returned {response.status}")

    def _profile_case(
        self,
        case: ParallelCase,
        port: int,
        served_name: str,
        container: str,
        profile_root: str,
    ) -> None:
        self.update(case, CaseStatus.PROFILE)
        base_url = f"http://127.0.0.1:{port}"
        self._post(f"{base_url}/start_profile")
        profiling = self.spec["profiling"]
        try:
            prompt = build_exact_prompt(
                int(profiling["input_tokens"]),
                count_tokens=lambda value: tokenize_count(
                    url=base_url,
                    model=served_name,
                    prompt=value,
                    timeout_seconds=30,
                ),
            )
            rows = run_workload(
                url=base_url,
                model=served_name,
                input_tokens=int(profiling["input_tokens"]),
                output_tokens=int(profiling["output_tokens"]),
                num_requests=int(profiling["num_requests"]),
                concurrency=int(profiling["concurrency"]),
                timeout_seconds=int(
                    self.spec["workload"]["request_timeout_seconds"]
                ),
                prompt=prompt,
            )
        finally:
            self._post(f"{base_url}/stop_profile")
        if any(not row["ok"] for row in rows):
            raise RuntimeError("profiling workload had failed requests")
        self.update(case, CaseStatus.ANALYZE)
        tool = (
            "/workspace/pipeline_parallel/profile_tool/profiler_collect.py"
        )
        completed = _docker_exec(
            container,
            [
                "python3",
                tool,
                "--profile-root",
                profile_root,
            ],
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                "profiler analysis failed: "
                + (completed.stderr or completed.stdout)[-2000:]
            )
        self._analyze_profile_metrics(case)

    def _analyze_profile_metrics(self, case: ParallelCase) -> None:
        profile_root = self.run_dir / "profiles" / case.case_id
        rank_metrics: list[dict[str, Any]] = []
        stages: list[dict[str, Any]] = []
        for rank_index, kernel_csv in enumerate(
            sorted(profile_root.glob("**/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
        ):
            summary = analyze_kernel_csv(kernel_csv)
            rank_metrics.append(
                {
                    "rank": rank_index,
                    "kernel_csv": str(kernel_csv),
                    **summary,
                }
            )
            duration = summary["duration_us"]
            stages.append(
                {
                    "stage": rank_index // max(1, case.tp),
                    "rank": rank_index,
                    "active_compute_us": duration["compute"],
                    "communication_us": duration["communication"],
                    "makespan_us": summary.get("measurement_makespan_us")
                    or sum(duration.values()),
                }
            )
        payload = {
            "case_id": case.case_id,
            "ranks": rank_metrics,
            "bubble": analyze_bubbles(stages) if stages else {},
        }
        (profile_root / "profile_metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _stop_owned_service(container: str, pid_file: str) -> None:
        command = (
            f"if [ -s {shlex.quote(pid_file)} ]; then "
            f"pid=$(cat {shlex.quote(pid_file)}); "
            'case "$pid" in (*[!0-9]*|"") exit 2;; esac; '
            'kill "$pid" 2>/dev/null || true; fi'
        )
        _docker_exec(container, ["bash", "-lc", command], capture_output=True)

    def _generate_report(self) -> None:
        rows: list[dict[str, Any]] = []
        for case in self.cases:
            path = self.run_dir / "cases" / case.case_id / "summary.json"
            if path.is_file():
                case_rows = json.loads(path.read_text(encoding="utf-8"))
                profile_path = (
                    self.run_dir
                    / "profiles"
                    / case.case_id
                    / "profile_metrics.json"
                )
                profile_summary: dict[str, Any] = {}
                if profile_path.is_file():
                    profile = json.loads(
                        profile_path.read_text(encoding="utf-8")
                    )
                    rank_values = profile.get("ranks", [])
                    if rank_values:
                        profile_summary = {
                            "compute_share": sum(
                                rank["share"]["compute"] for rank in rank_values
                            )
                            / len(rank_values),
                            "memory_share": sum(
                                rank["share"]["memory"] for rank in rank_values
                            )
                            / len(rank_values),
                            "communication_share": sum(
                                rank["share"]["communication"]
                                for rank in rank_values
                            )
                            / len(rank_values),
                            "pp_p2p_us": sum(
                                rank["communication"]
                                .get("send_recv", {})
                                .get("total_us", 0)
                                for rank in rank_values
                            ),
                            "all_reduce_us": sum(
                                rank["communication"]
                                .get("all_reduce", {})
                                .get("total_us", 0)
                                for rank in rank_values
                            ),
                            "all_gather_us": sum(
                                rank["communication"]
                                .get("all_gather", {})
                                .get("total_us", 0)
                                for rank in rank_values
                            ),
                            "reduce_scatter_us": sum(
                                rank["communication"]
                                .get("reduce_scatter", {})
                                .get("total_us", 0)
                                for rank in rank_values
                            ),
                            "all_to_all_us": sum(
                                rank["communication"]
                                .get("all_to_all", {})
                                .get("total_us", 0)
                                for rank in rank_values
                            ),
                        }
                    bubble = profile.get("bubble", {})
                    stages = bubble.get("stages", [])
                    if stages:
                        profile_summary.update(
                            {
                                "bubble_ratio": sum(
                                    stage["bubble_ratio"] for stage in stages
                                )
                                / len(stages),
                                "active_imbalance_ratio": bubble.get(
                                    "active_imbalance_ratio", 0
                                ),
                            }
                        )
                rows.extend(
                    [{**row, **profile_summary} for row in case_rows]
                )
        # Report each workload point/repetition; a later analysis pass can group
        # medians without losing raw observations.
        generate_report(
            self.run_dir,
            rows,
            metadata={
                "run_id": self.run_dir.name,
                "model": self.spec["model"]["name"],
            },
        )

    def _cancel_requested(self) -> bool:
        return (self.run_dir / "CANCEL").exists()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    return RemoteController(args.run_dir, args.spec).run()


if __name__ == "__main__":
    raise SystemExit(main())
