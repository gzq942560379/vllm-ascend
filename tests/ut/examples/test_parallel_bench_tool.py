# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for the recoverable vLLM-Ascend parallel benchmark tool."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TOOL_DIR = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "pipeline_parallel"
    / "profile_tool"
)
WINDOWS_LAUNCHER = TOOL_DIR / "parallel_bench.ps1"
DOCKER_SCRIPT = TOOL_DIR.parent / "docker.sh"
REMOTE_CONTROLLER = TOOL_DIR / "benchmark_remote.py"
STANDALONE_REMOTE_PROFILER = TOOL_DIR / "profile_remote.py"
PROFILE_SCOPE_INSTALLER = TOOL_DIR / "install_profile_scopes.py"
MODEL_RUNNER = (
    Path(__file__).resolve().parents[3]
    / "vllm_ascend"
    / "worker"
    / "model_runner_v1.py"
)
DEFAULT_CLIENT_CONFIG = TOOL_DIR / "configs" / "parallel_bench_config.json"
SMOKE_CONFIG = TOOL_DIR / "configs" / "qwen3_30b_a3b_smoke.json"
QWEN3_8B_SMOKE_CONFIG = TOOL_DIR / "configs" / "qwen3_8b_smoke.json"
sys.path.insert(0, str(TOOL_DIR))

from benchmark_remote import (  # noqa: E402
    CaseStatus,
    ExperimentState,
    MERGED_TRACE_NAME,
    merge_trace_view_files,
    runnable_case_ids,
)
from install_profile_scopes import (  # noqa: E402
    patch_envs_source,
    patch_runner_source,
)
from experiment_schema import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    PROFILE_SCOPE_NAMES,
    CapabilitySet,
    default_spec,
    enabled_profile_scopes,
    expand_boundary_matrix,
    expand_cases,
    expand_quick_matrix,
    render_parallel_flags,
    validate_spec,
)
from metrics_analyzer import (  # noqa: E402
    analyze_bubbles,
    classify_kernel,
    suggest_contiguous_stage_boundaries,
    summarize_kernel_rows,
)
from report_generator import generate_report  # noqa: E402
from resource_scheduler import DeviceInfo, select_devices  # noqa: E402
from workload_streaming import (  # noqa: E402
    RequestTimeline,
    build_exact_prompt,
    percentile,
)


class WindowsLauncherContractTests(unittest.TestCase):

    def test_editable_json_controls_connection_run_and_output(self) -> None:
        config = json.loads(DEFAULT_CLIENT_CONFIG.read_text(encoding="utf-8"))

        self.assertEqual(config["client"]["server"], "192.168.13.190")
        self.assertEqual(config["client"]["ssh_user"], "root")
        self.assertEqual(config["client"]["ssh_port"], 22)
        self.assertEqual(
            config["client"]["output_dir"], r"D:\vllm-parallel-bench"
        )
        self.assertEqual(
            config["client"]["remote_project"],
            "/home/vllm/l00977701/pipeline_parallel",
        )
        self.assertEqual(config["run"]["run_id"], "auto")
        self.assertIn("workload", config)
        self.assertIn("profiling", config)

    def test_launcher_supports_one_config_file_for_every_action(self) -> None:
        launcher = WINDOWS_LAUNCHER.read_text(encoding="utf-8")

        self.assertIn('[string]$ConfigFile =', launcher)
        self.assertIn("Resolve-RunId", launcher)
        self.assertIn(".last-run.json", launcher)
        self.assertIn('Write-Host "Config:', launcher)
        self.assertNotIn("-RunId is required for $Action", launcher)

    def test_launcher_batches_windows_submit_without_ssh_multiplexing(self) -> None:
        launcher = WINDOWS_LAUNCHER.read_text(encoding="utf-8")
        smoke_config = json.loads(
            QWEN3_8B_SMOKE_CONFIG.read_text(encoding="utf-8")
        )

        self.assertIn("ControlMaster=auto", launcher)
        self.assertIn("ControlPersist=", launcher)
        self.assertIn("ControlPath=", launcher)
        self.assertIn("Open-SshMaster", launcher)
        self.assertIn("Close-SshMaster", launcher)
        self.assertIn('$env:OS -eq "Windows_NT"', launcher)
        self.assertIn("Win32 OpenSSH", launcher)
        self.assertIn("$temporaryArchive", launcher)
        self.assertIn("uploading bundled benchmark payload", launcher)
        self.assertIn("password prompt 1/2", launcher)
        self.assertIn("password prompt 2/2", launcher)
        self.assertFalse(smoke_config["client"]["ssh_multiplexing"])
        self.assertEqual(
            smoke_config["client"]["ssh_connect_timeout_seconds"], 15
        )
        self.assertEqual(
            smoke_config["client"]["remote_command_timeout_seconds"], 600
        )

    def test_launcher_times_out_failed_connections_and_remote_steps(self) -> None:
        launcher = WINDOWS_LAUNCHER.read_text(encoding="utf-8")

        self.assertIn("ConnectTimeout=", launcher)
        self.assertIn("ServerAliveInterval=", launcher)
        self.assertIn("ServerAliveCountMax=", launcher)
        self.assertIn("timeout --foreground", launcher)
        self.assertIn("timed out after", launcher)
        self.assertIn('-Operation "running remote submit workflow"', launcher)

    def test_fetch_packages_only_merged_profile_traces(self) -> None:
        launcher = WINDOWS_LAUNCHER.read_text(encoding="utf-8")

        self.assertIn("-path '$RunId/profiles' -prune", launcher)
        self.assertIn("-name 'merged_trace_view.json'", launcher)
        self.assertIn("tar --null --files-from=", launcher)
        self.assertIn("sleep 3600; rm -f $remoteFetchArchive", launcher)
        self.assertIn("Profiles: merged_trace_view.json only", launcher)
        self.assertNotIn(
            'Invoke-Scp -Arguments @("-r", "${target}:$remoteRun"',
            launcher,
        )

    def test_launcher_encodes_remote_commands_before_ssh(self) -> None:
        launcher = WINDOWS_LAUNCHER.read_text(encoding="utf-8")

        self.assertIn("[Convert]::ToBase64String", launcher)
        self.assertIn("| base64 -d |", launcher)
        self.assertIn("timeout --foreground ${TimeoutSeconds}s bash", launcher)

    def test_launcher_accepts_custom_matrix_from_json(self) -> None:
        launcher = WINDOWS_LAUNCHER.read_text(encoding="utf-8")

        self.assertIn(
            '[ValidateSet("quick", "boundary", "custom")]', launcher
        )

    def test_submit_mounts_host_model_at_container_models(self) -> None:
        launcher = WINDOWS_LAUNCHER.read_text(encoding="utf-8")
        docker_script = DOCKER_SCRIPT.read_text(encoding="utf-8")
        controller = REMOTE_CONTROLLER.read_text(encoding="utf-8")

        self.assertIn('MODEL_DIR=\'$Model\'', launcher)
        self.assertIn("MODEL_DIR='$Model' '$remoteDocker' restart", launcher)
        self.assertIn('-v "${MODEL_DIR}:/models"', docker_script)
        self.assertIn('SERVE_MODEL="${SERVE_MODEL:-/models}"', docker_script)
        self.assertIn("cmd_restart() {\n  validate_model_dir", docker_script)
        self.assertIn('str(model["container_path"])', controller)

    def test_launcher_normalizes_shell_scripts_before_upload(self) -> None:
        launcher = WINDOWS_LAUNCHER.read_text(encoding="utf-8")

        self.assertIn("function Write-LfCopy", launcher)
        self.assertIn('.Replace("`r`n", "`n")', launcher)
        self.assertIn("$temporaryDocker", launcher)
        self.assertIn("$temporarySelector", launcher)

    @unittest.skipUnless(shutil.which("powershell"), "PowerShell is required")
    def test_smoke_custom_config_completes_dry_run(self) -> None:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(WINDOWS_LAUNCHER),
                "-Action",
                "Submit",
                "-ConfigFile",
                str(QWEN3_8B_SMOKE_CONFIG),
                "-DryRun",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('"preset":  "custom"', completed.stdout)
        self.assertIn('"case_id":  "SMOKE_P1"', completed.stdout)

    @unittest.skipUnless(
        os.name == "nt" and shutil.which("powershell"),
        "Windows PowerShell is required",
    )
    def test_windows_submit_uses_one_scp_and_one_ssh_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mock_bin = root / "bin"
            mock_bin.mkdir()
            ssh_log = root / "ssh.log"
            scp_log = root / "scp.log"
            (mock_bin / "ssh.cmd").write_text(
                '@echo off\r\necho %*>>"%VPB_SSH_LOG%"\r\nexit /b 0\r\n',
                encoding="utf-8",
            )
            (mock_bin / "scp.cmd").write_text(
                '@echo off\r\necho %*>>"%VPB_SCP_LOG%"\r\nexit /b 0\r\n',
                encoding="utf-8",
            )
            config = json.loads(
                QWEN3_8B_SMOKE_CONFIG.read_text(encoding="utf-8")
            )
            config["client"]["output_dir"] = str(root / "output")
            config_path = root / "qwen3_8b_smoke.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            environment = os.environ.copy()
            environment["PATH"] = str(mock_bin) + os.pathsep + environment["PATH"]
            environment["VPB_SSH_LOG"] = str(ssh_log)
            environment["VPB_SCP_LOG"] = str(scp_log)

            completed = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(WINDOWS_LAUNCHER),
                    "-Action",
                    "Submit",
                    "-ConfigFile",
                    str(config_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(len(scp_log.read_text().splitlines()), 1)
            self.assertEqual(len(ssh_log.read_text().splitlines()), 1)
            self.assertIn("password prompt 1/2", completed.stdout)
            self.assertIn("password prompt 2/2", completed.stdout)


class HighLevelProfileScopeContractTests(unittest.TestCase):

    def test_submit_installs_scopes_into_the_image_runtime(self) -> None:
        launcher = WINDOWS_LAUNCHER.read_text(encoding="utf-8")

        self.assertIn('"install_profile_scopes.py"', launcher)
        self.assertIn(
            "docker exec '$Container' python3 ", launcher
        )
        self.assertIn(
            "/workspace/pipeline_parallel/profile_tool/"
            "install_profile_scopes.py",
            launcher,
        )

    def test_runtime_installer_patches_old_image_sources(self) -> None:
        envs_source = '''import os
env_variables = {
    "EXISTING": lambda: 1,
}
# end-env-vars-definition
'''
        runner_source = '''from contextlib import nullcontext
from typing import NamedTuple
from vllm_ascend.ascend_config import get_ascend_config

class ExecuteModelState(NamedTuple):
    value: int

def execute(self):
    with (
            record_function_or_nullcontext("forward"),
    ):
        pass
'''

        patched_envs = patch_envs_source(envs_source)
        patched_runner = patch_runner_source(runner_source)

        self.assertIn("VLLM_ASCEND_PROFILING_SCOPES", patched_envs)
        self.assertIn("from vllm_ascend import envs as envs_ascend", patched_runner)
        self.assertIn("_forward_profile_scope(self.attn_state)", patched_runner)
        compile(patched_envs, "envs.py", "exec")
        compile(patched_runner, "model_runner_v1.py", "exec")

    def test_parallel_launcher_passes_only_configured_scopes(self) -> None:
        source = REMOTE_CONTROLLER.read_text(encoding="utf-8")

        self.assertIn("VLLM_CUSTOM_SCOPES_FOR_PROFILING=1", source)
        self.assertIn("VLLM_ASCEND_PROFILING_SCOPES=", source)
        self.assertIn("enabled_profile_scopes(self.spec)", source)

    def test_model_forward_has_nested_semantic_scopes(self) -> None:
        source = MODEL_RUNNER.read_text(encoding="utf-8")

        self.assertIn('_forward_profile_context("forward")', source)
        self.assertIn("_forward_profile_scope(self.attn_state)", source)
        self.assertIn("ENABLED_FORWARD_PROFILE_SCOPES", source)
        for scope in (
            'return "prefill"',
            'return "decode"',
            'return "chunked_prefill"',
            'return "spec_decode"',
        ):
            self.assertIn(scope, source)

    def test_8b_smoke_keeps_only_forward_cpu_scope(self) -> None:
        config = json.loads(
            QWEN3_8B_SMOKE_CONFIG.read_text(encoding="utf-8")
        )

        self.assertEqual(config["profiling"]["trace_mode"], "scopes_only")
        enabled = [
            name
            for name, value in config["profiling"]["scopes"].items()
            if value
        ]
        self.assertEqual(enabled, ["forward"])
        self.assertIn("NPU MEM", config["profiling"]["exclude_tracks"])
        self.assertIn("QoS", config["profiling"]["exclude_tracks"])


class ExperimentSchemaTests(unittest.TestCase):

    def test_profile_scopes_are_independently_configurable(self) -> None:
        spec = default_spec()
        spec["profiling"]["scopes"] = {
            name: name in {"prefill", "decode"}
            for name in PROFILE_SCOPE_NAMES
        }

        validate_spec(spec)

        self.assertEqual(enabled_profile_scopes(spec), ("prefill", "decode"))

    def test_profile_scopes_reject_unknown_names_and_non_booleans(self) -> None:
        spec = default_spec()
        spec["profiling"]["scopes"]["typo"] = True
        with self.assertRaisesRegex(ValueError, "unsupported names: typo"):
            validate_spec(spec)

        spec = default_spec()
        spec["profiling"]["scopes"]["forward"] = "yes"
        with self.assertRaisesRegex(ValueError, "forward.*true or false"):
            validate_spec(spec)

    def test_profile_trace_mode_rejects_unknown_values(self) -> None:
        spec = default_spec()
        spec["profiling"]["trace_mode"] = "operators_off"

        with self.assertRaisesRegex(ValueError, "profiling.trace_mode"):
            validate_spec(spec)

    def test_profile_excluded_tracks_require_non_empty_names(self) -> None:
        spec = default_spec()
        spec["profiling"]["exclude_tracks"] = ["HBM", ""]

        with self.assertRaisesRegex(ValueError, "exclude_tracks"):
            validate_spec(spec)

    def test_custom_matrix_requires_explicit_cases(self) -> None:
        spec = default_spec()
        spec["matrix"] = {"preset": "custom", "cases": []}

        with self.assertRaisesRegex(ValueError, "custom.*case"):
            expand_cases(spec)

    def test_custom_matrix_uses_explicit_cases(self) -> None:
        spec = default_spec()
        spec["matrix"] = {
            "preset": "custom",
            "cases": [
                {
                    "case_id": "SMOKE_P1",
                    "pp": 2,
                    "tp": 1,
                    "ep": False,
                    "cp": 1,
                    "profile": False,
                }
            ],
        }

        cases = expand_cases(spec)

        self.assertEqual([case.case_id for case in cases], ["SMOKE_P1"])
        self.assertEqual(cases[0].world_size, 2)

    def test_default_spec_and_quick_matrix(self) -> None:
        spec = default_spec()
        self.assertEqual(spec["model"]["path"], DEFAULT_MODEL_PATH)
        self.assertEqual(spec["model"]["container_path"], "/models")
        self.assertEqual(spec["model"]["kind"], "moe")
        self.assertEqual(spec["resource"]["max_wait_seconds"], 6 * 60 * 60)

        cases = expand_quick_matrix(spec)
        self.assertEqual(
            [case.case_id for case in cases],
            ["R0", "P1", "T1", "E1", "T2", "E2"],
        )
        self.assertEqual(
            [(case.pp, case.tp, case.ep, case.world_size) for case in cases],
            [
                (1, 1, False, 1),
                (2, 1, False, 2),
                (2, 2, False, 4),
                (2, 2, True, 4),
                (4, 2, False, 8),
                (4, 2, True, 8),
            ],
        )

    def test_boundary_matrix_never_exceeds_sixteen_dies(self) -> None:
        cases = expand_boundary_matrix(default_spec())
        self.assertTrue(cases)
        self.assertTrue(all(case.world_size <= 16 for case in cases))
        self.assertTrue(any(case.pp == 8 and case.tp == 2 for case in cases))

    def test_dense_model_quick_matrix_skips_ep(self) -> None:
        spec = default_spec()
        spec["model"]["kind"] = "dense"
        cases = expand_quick_matrix(spec)
        self.assertEqual(
            [case.case_id for case in cases], ["R0", "P1", "T1", "T2"]
        )

    def test_capability_adapter_skips_unsupported_cp(self) -> None:
        capabilities = CapabilitySet.from_help_text(
            "vllm serve ... --enable-expert-parallel"
        )
        decision = render_parallel_flags(
            pp=2,
            tp=2,
            ep=True,
            cp=2,
            capabilities=capabilities,
        )
        self.assertEqual(decision.status, "SKIPPED_UNSUPPORTED")
        self.assertIn("context parallel", decision.reason.lower())

        supported = CapabilitySet.from_help_text(
            "--enable-expert-parallel --context-parallel-size"
        )
        decision = render_parallel_flags(
            pp=2,
            tp=2,
            ep=True,
            cp=2,
            capabilities=supported,
        )
        self.assertIsNone(decision.status)
        self.assertIn("--pipeline-parallel-size", decision.flags)
        self.assertIn("--enable-expert-parallel", decision.flags)
        self.assertIn("--context-parallel-size", decision.flags)


class StreamingMetricTests(unittest.TestCase):

    def test_true_streaming_ttft_tpot_and_itl(self) -> None:
        timeline = RequestTimeline(
            started_at=10.0,
            token_timestamps=(10.2, 10.3, 10.5, 10.8),
            completed_at=10.81,
            prompt_tokens=128,
            output_tokens=4,
        )
        metrics = timeline.metrics()
        self.assertAlmostEqual(metrics["ttft_ms"], 200.0)
        self.assertAlmostEqual(metrics["tpot_ms"], 203.3333333333)
        self.assertAlmostEqual(metrics["itl_p50_ms"], 200.0)
        self.assertAlmostEqual(metrics["itl_p95_ms"], 290.0)
        self.assertAlmostEqual(metrics["e2e_ms"], 810.0)

    def test_percentile_and_single_token_tpot(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0], 50), 2.0)
        metrics = RequestTimeline(
            started_at=1.0,
            token_timestamps=(1.4,),
            completed_at=1.5,
            prompt_tokens=8,
            output_tokens=1,
        ).metrics()
        self.assertIsNone(metrics["tpot_ms"])
        self.assertIsNone(metrics["itl_p95_ms"])

    def test_exact_prompt_builder_hits_requested_token_count(self) -> None:
        prompt = build_exact_prompt(128, count_tokens=len)
        self.assertEqual(len(prompt), 128)
        self.assertIn("流水线", prompt)


class ResourceSchedulerTests(unittest.TestCase):

    def test_topology_aware_selection_prefers_complete_cards(self) -> None:
        devices = [
            DeviceInfo(logic_id=0, card_id=0, chip_id=0),
            DeviceInfo(logic_id=1, card_id=0, chip_id=1),
            DeviceInfo(logic_id=2, card_id=1, chip_id=0),
            DeviceInfo(logic_id=3, card_id=1, chip_id=1),
            DeviceInfo(logic_id=4, card_id=2, chip_id=0),
        ]
        selected = select_devices(devices, count=4, topology_aware=True)
        self.assertEqual([device.logic_id for device in selected], [0, 1, 2, 3])
        self.assertEqual(len({device.logic_id for device in selected}), 4)

    def test_insufficient_devices_returns_empty_selection(self) -> None:
        devices = [DeviceInfo(logic_id=0, card_id=0, chip_id=0)]
        self.assertEqual(select_devices(devices, count=2), [])


class ResumeStateTests(unittest.TestCase):

    def test_resume_does_not_repeat_terminal_cases(self) -> None:
        state = ExperimentState.new("run-1", ["P1", "T1", "E1"])
        state.cases["P1"].status = CaseStatus.COMPLETE
        state.cases["T1"].status = CaseStatus.SKIPPED_CAPACITY
        state.cases["E1"].status = CaseStatus.RETRYABLE
        self.assertEqual(runnable_case_ids(state), ["E1"])

    def test_state_save_is_readable_and_preserves_attempts(self) -> None:
        state = ExperimentState.new("run-2", ["P1"])
        state.cases["P1"].attempts = 2
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            state.save(path)
            loaded = ExperimentState.load(path)
        self.assertEqual(loaded.run_id, "run-2")
        self.assertEqual(loaded.cases["P1"].attempts, 2)


class TraceMergeTests(unittest.TestCase):

    def test_rank_trace_arrays_are_streamed_into_one_valid_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile_root = Path(temporary)
            rank0 = profile_root / "rank0" / "ASCEND_PROFILER_OUTPUT"
            rank1 = profile_root / "rank1" / "ASCEND_PROFILER_OUTPUT"
            rank0.mkdir(parents=True)
            rank1.mkdir(parents=True)
            (rank0 / "trace_view.json").write_text(
                '[{"pid": 10, "name": "rank0-event"}]\n',
                encoding="utf-8",
            )
            (rank1 / "trace_view.json").write_text(
                '[\n  {"pid": 20, "name": "rank1-event"}\n]\n',
                encoding="utf-8",
            )

            merged_path = merge_trace_view_files(
                profile_root, expected_minimum=2, chunk_size=7
            )

            self.assertEqual(merged_path.name, MERGED_TRACE_NAME)
            self.assertEqual(
                json.loads(merged_path.read_text(encoding="utf-8")),
                [
                    {"pid": 10, "name": "rank0-event"},
                    {"pid": 20, "name": "rank1-event"},
                ],
            )

    def test_merge_fails_when_a_parallel_rank_trace_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile_root = Path(temporary)
            rank0 = profile_root / "rank0" / "ASCEND_PROFILER_OUTPUT"
            rank0.mkdir(parents=True)
            (rank0 / "trace_view.json").write_text("[]", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "expected at least 2"):
                merge_trace_view_files(profile_root, expected_minimum=2)

    def test_scopes_only_merge_removes_cpu_and_runtime_details(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile_root = Path(temporary)
            events_by_rank = (
                [
                    {
                        "pid": 10,
                        "cat": "cpu_op",
                        "name": "forward",
                        "args": {"text": "comma, brace } and quote \""},
                    },
                    {"pid": 10, "cat": "cpu_op", "name": "aten::slice"},
                    {"pid": 10, "cat": "enqueue", "name": "Enqueue@Matmul"},
                    {
                        "pid": 110,
                        "ph": "M",
                        "name": "process_name",
                        "args": {"name": "HBM"},
                    },
                    {"pid": 110, "ph": "C", "name": "HBM 0/Read"},
                    {"pid": 100, "cat": "", "name": "MatMulV3"},
                ],
                [
                    {"pid": 20, "cat": "cpu_op", "name": "prefill"},
                    {"pid": 20, "cat": "dequeue", "name": "Dequeue@Matmul"},
                    {"pid": 200, "cat": "", "name": "HcclSend"},
                ],
            )
            for rank, events in enumerate(events_by_rank):
                output = (
                    profile_root
                    / f"rank{rank}"
                    / "ASCEND_PROFILER_OUTPUT"
                )
                output.mkdir(parents=True)
                (output / "trace_view.json").write_text(
                    json.dumps(events), encoding="utf-8"
                )

            merged_path = merge_trace_view_files(
                profile_root,
                expected_minimum=2,
                chunk_size=5,
                allowed_cpu_scopes=frozenset({"forward"}),
                excluded_process_names=frozenset({"HBM"}),
            )
            merged = json.loads(merged_path.read_text(encoding="utf-8"))

            self.assertEqual(
                [event["name"] for event in merged],
                ["forward", "MatMulV3", "HcclSend"],
            )

    def test_controller_merges_only_pipeline_parallel_profiles(self) -> None:
        controller = REMOTE_CONTROLLER.read_text(encoding="utf-8")

        self.assertIn("if case.pp > 1:", controller)
        self.assertIn("expected_minimum=case.world_size", controller)


class TraceAnalysisTests(unittest.TestCase):

    def test_kernel_classification_and_time_shares(self) -> None:
        rows = [
            {"Name": "aclnnMatmul", "Duration(us)": "100"},
            {"Name": "aclnnInplaceCopy", "Duration(us)": "20"},
            {"Name": "HcomAllReduce", "Duration(us)": "30"},
            {"Name": "aclnnAdd", "Duration(us)": "50"},
        ]
        self.assertEqual(classify_kernel("HcomAllToAllV"), "communication")
        summary = summarize_kernel_rows(rows)
        self.assertEqual(summary["duration_us"]["compute"], 150.0)
        self.assertEqual(summary["duration_us"]["memory"], 20.0)
        self.assertEqual(summary["duration_us"]["communication"], 30.0)
        self.assertAlmostEqual(summary["share"]["compute"], 0.75)
        self.assertEqual(summary["communication"]["all_reduce"]["count"], 1)

    def test_stage_bubble_and_imbalance(self) -> None:
        summary = analyze_bubbles(
            [
                {
                    "stage": 0,
                    "active_compute_us": 800.0,
                    "communication_us": 100.0,
                    "makespan_us": 1000.0,
                },
                {
                    "stage": 1,
                    "active_compute_us": 500.0,
                    "communication_us": 200.0,
                    "makespan_us": 1000.0,
                },
            ]
        )
        self.assertAlmostEqual(summary["stages"][0]["bubble_ratio"], 0.2)
        self.assertAlmostEqual(summary["stages"][1]["bubble_ratio"], 0.5)
        self.assertAlmostEqual(summary["active_imbalance_ratio"], 1.6)

    def test_weighted_pp_partition_keeps_heavy_layers_separate(self) -> None:
        suggestion = suggest_contiguous_stage_boundaries(
            [1.0, 1.0, 8.0, 1.0, 1.0], stages=2
        )
        self.assertEqual(suggestion["boundaries"], [0, 2, 5])
        self.assertEqual(suggestion["stage_costs"], [2.0, 10.0])
        self.assertEqual(suggestion["max_stage_cost"], 10.0)


class ReportTests(unittest.TestCase):

    def test_report_and_required_svg_charts_are_generated(self) -> None:
        rows = [
            {
                "case_id": "P1",
                "pp": 2,
                "tp": 1,
                "ep": False,
                "concurrency": 4,
                "output_throughput": 40.0,
                "ttft_p50_ms": 100.0,
                "tpot_p50_ms": 20.0,
                "bubble_ratio": 0.55,
                "active_imbalance_ratio": 1.8,
                "compute_share": 0.45,
                "memory_share": 0.15,
                "communication_share": 0.4,
                "pp_p2p_us": 2000.0,
            },
            {
                "case_id": "T1",
                "pp": 2,
                "tp": 2,
                "ep": False,
                "concurrency": 4,
                "output_throughput": 70.0,
                "ttft_p50_ms": 120.0,
                "tpot_p50_ms": 13.0,
                "bubble_ratio": 0.3,
                "active_imbalance_ratio": 1.1,
                "compute_share": 0.6,
                "memory_share": 0.1,
                "communication_share": 0.3,
                "pp_p2p_us": 800.0,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = generate_report(
                output,
                rows,
                metadata={"run_id": "demo", "model": "Qwen3-30B-A3B"},
            )
            self.assertTrue((output / "report.md").is_file())
            self.assertTrue((output / "report.html").is_file())
            self.assertTrue((output / "summary.csv").is_file())
            for name in (
                "throughput_scaling.svg",
                "ttft_tpot_comparison.svg",
                "bubble_by_stage.svg",
                "compute_memory_comm_share.svg",
                "communication_breakdown.svg",
                "scaling_efficiency.svg",
            ):
                self.assertTrue((output / "charts" / name).is_file(), name)
            report = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("PP 流水线优化建议", report)
            self.assertIn("stage", report.lower())
            self.assertIn("report_md", result)


if __name__ == "__main__":
    unittest.main()
