# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for the recoverable vLLM-Ascend parallel benchmark tool."""

from __future__ import annotations

import json
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
DEFAULT_CLIENT_CONFIG = TOOL_DIR / "configs" / "parallel_bench_config.json"
sys.path.insert(0, str(TOOL_DIR))

from benchmark_remote import (  # noqa: E402
    CaseStatus,
    ExperimentState,
    runnable_case_ids,
)
from experiment_schema import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    CapabilitySet,
    default_spec,
    expand_boundary_matrix,
    expand_cases,
    expand_quick_matrix,
    render_parallel_flags,
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

    def test_launcher_reuses_one_authenticated_ssh_connection(self) -> None:
        launcher = WINDOWS_LAUNCHER.read_text(encoding="utf-8")

        self.assertIn("ControlMaster=auto", launcher)
        self.assertIn("ControlPersist=", launcher)
        self.assertIn("ControlPath=", launcher)
        self.assertIn("Open-SshMaster", launcher)
        self.assertIn("Close-SshMaster", launcher)

    def test_launcher_accepts_custom_matrix_from_json(self) -> None:
        launcher = WINDOWS_LAUNCHER.read_text(encoding="utf-8")

        self.assertIn(
            '[ValidateSet("quick", "boundary", "custom")]', launcher
        )


class ExperimentSchemaTests(unittest.TestCase):

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
