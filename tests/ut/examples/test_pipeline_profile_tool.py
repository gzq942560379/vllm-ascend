# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
import tempfile
import unittest
from base64 import b64encode
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch


TOOL_PATH = (
    Path(__file__).parents[3]
    / "examples"
    / "pipeline_parallel"
    / "profile_tool"
    / "profile_remote.py"
)


def _load_tool():
    spec = importlib.util.spec_from_file_location("profile_remote", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProfileToolTest(unittest.TestCase):
    def test_profile_config_rejects_unsafe_or_invalid_values(self):
        tool = _load_tool()

        with self.assertRaisesRegex(ValueError, "container"):
            tool.ProfileConfig.from_mapping(
                {
                    "container": "qwen;docker rm",
                    "model": "/models/Qwen3-8B",
                    "device": "0",
                }
            )

        with self.assertRaisesRegex(ValueError, "device"):
            tool.ProfileConfig.from_mapping(
                {
                    "container": "qwen3_parallel_nightly",
                    "model": "/models/Qwen3-8B",
                    "device": "0,1",
                }
            )

        with self.assertRaisesRegex(ValueError, "port"):
            tool.ProfileConfig.from_mapping(
                {
                    "container": "qwen3_parallel_nightly",
                    "model": "/models/Qwen3-8B",
                    "device": "0",
                    "port": 70000,
                }
            )

        with self.assertRaisesRegex(ValueError, "execution_mode"):
            tool.ProfileConfig.from_mapping(
                {
                    "container": "qwen3_parallel_nightly",
                    "model": "/models/Qwen3-8B",
                    "device": "0",
                    "execution_mode": "unknown",
                }
            )

        with self.assertRaisesRegex(ValueError, "concurrency"):
            tool.ProfileConfig.from_mapping(
                {
                    "container": "qwen3_parallel_nightly",
                    "model": "/models/Qwen3-8B",
                    "device": "0",
                    "num_requests": 4,
                    "concurrency": 8,
                }
            )

    def test_build_vllm_command_enables_single_request_npu_profiling(self):
        tool = _load_tool()
        config = tool.ProfileConfig.from_mapping(
            {
                "container": "qwen3_parallel_nightly",
                "model": "/models/Qwen3-8B",
                "device": "0",
                "port": 8010,
                "max_tokens": 16,
            }
        )

        command = tool.build_vllm_command(
            config,
            served_model_name="qwen3-8b-profile-20260730",
            profile_dir="/workspace/profile_tool_runs/20260730/raw",
        )

        self.assertEqual(command[:3], ["vllm", "serve", "/models/Qwen3-8B"])
        self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "1")
        self.assertEqual(command[command.index("--pipeline-parallel-size") + 1], "1")
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "1")
        self.assertEqual(command[command.index("--port") + 1], "8010")
        self.assertIn("--enforce-eager", command)
        self.assertEqual(
            command[command.index("--kv-cache-memory-bytes") + 1],
            str(4 * 1024**3),
        )

        profiler_config = json.loads(
            command[command.index("--profiler-config") + 1]
        )
        self.assertEqual(
            profiler_config,
            {
                "profiler": "torch",
                "torch_profiler_dir": "/workspace/profile_tool_runs/20260730/raw",
                "torch_profiler_record_shapes": True,
                "torch_profiler_with_memory": True,
                "torch_profiler_with_stack": False,
            },
        )

    def test_aclgraph_and_multi_request_settings_change_the_serve_plan(self):
        tool = _load_tool()
        config = tool.ProfileConfig.from_mapping(
            {
                "container": "qwen3_parallel_nightly",
                "model": "/models/Qwen3-8B",
                "device": "0",
                "execution_mode": "aclgraph",
                "num_requests": 16,
                "concurrency": 4,
            }
        )

        command = tool.build_vllm_command(
            config,
            served_model_name="qwen3-8b-profile-multi",
            profile_dir="/workspace/profile_tool_runs/multi/raw",
        )

        self.assertNotIn("--enforce-eager", command)
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "4")
        request = tool.build_chat_request(config, "qwen3-8b-profile-multi")
        self.assertEqual(request["model"], "qwen3-8b-profile-multi")
        self.assertEqual(request["max_tokens"], 16)

    def test_expected_vllm_version_accepts_build_suffix_and_rejects_mismatch(self):
        tool = _load_tool()

        tool.validate_vllm_version("0.25.1+ascend", "0.25.1")
        tool.validate_vllm_version("0.25.1", "")
        with self.assertRaisesRegex(RuntimeError, "vLLM version mismatch"):
            tool.validate_vllm_version("0.25.1", "0.24.0")

    def test_find_trace_view_requires_exactly_one_result(self):
        tool = _load_tool()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "rank0_1_ascend_pt" / "ASCEND_PROFILER_OUTPUT"
            output.mkdir(parents=True)
            expected = output / "trace_view.json"
            expected.write_text("{}", encoding="utf-8")

            self.assertEqual(tool.find_trace_view(root), expected)

            second = root / "rank0_2_ascend_pt" / "ASCEND_PROFILER_OUTPUT"
            second.mkdir(parents=True)
            (second / "trace_view.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                tool.find_trace_view(root)

    def test_decode_config_preserves_a_real_unicode_prompt(self):
        tool = _load_tool()
        values = {
            "container": "qwen3_parallel_nightly",
            "model": "/models/Qwen3-8B",
            "device": "0",
            "prompt": "请解释流水线并行。",
        }
        encoded = b64encode(
            json.dumps(values, ensure_ascii=False).encode("utf-8")
        ).decode("ascii")

        config = tool._decode_config(encoded)

        self.assertEqual(config.prompt, values["prompt"])
        self.assertEqual(config.port, 8010)

    def test_run_profile_executes_and_cleans_up_the_full_control_flow(self):
        tool = _load_tool()
        config = tool.ProfileConfig.from_mapping(
            {
                "container": "qwen3_parallel_nightly",
                "model": "/models/Qwen3-8B",
                "device": "0",
                "num_requests": 3,
                "concurrency": 2,
            }
        )
        http_calls = []

        def fake_http(_config, method, path, payload=None, check=True):
            http_calls.append((method, path, payload, check))
            if len(http_calls) == 1:
                return CompletedProcess([], 1, "", "connection refused")
            if path == "/v1/chat/completions":
                return CompletedProcess(
                    [],
                    0,
                    '{"choices":[{"message":{"content":"ok"}}]}',
                    "",
                )
            return CompletedProcess([], 0, "", "")

        batch_result = json.dumps(
            {
                "num_requests": 3,
                "concurrency": 2,
                "successful": 3,
                "failed": 0,
                "responses": [],
            }
        )

        def fake_docker_exec(_container, command, **_kwargs):
            if len(command) >= 3 and command[:2] == ["bash", "-lc"]:
                if "find " in command[2]:
                    return CompletedProcess(
                        [],
                        0,
                        "/workspace/profile_tool_runs/run/rank0_1_ascend_pt\n",
                        "",
                    )
            return CompletedProcess([], 0, "", "")

        def fake_run(command, **_kwargs):
            if command[:2] == ["docker", "cp"]:
                destination = Path(command[-1])
                destination.write_text("copied", encoding="utf-8")
            return CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(tool.Path, "home", return_value=Path(temp_dir)),
                patch.object(tool, "_run", side_effect=fake_run),
                patch.object(tool, "_docker_exec", side_effect=fake_docker_exec),
                patch.object(tool, "_container_http", side_effect=fake_http),
                patch.object(
                    tool,
                    "_container_chat_batch",
                    return_value=CompletedProcess([], 0, batch_result, ""),
                ) as chat_batch,
                patch.object(tool, "_wait_for_health"),
                patch.object(
                    tool,
                    "_get_runtime_versions",
                    return_value={"vllm": "0.25.1"},
                ),
                patch.object(
                    tool,
                    "_analyse_in_container",
                    return_value=(
                        "/workspace/profile_tool_runs/run/raw/"
                        "rank0_1_ascend_pt/ASCEND_PROFILER_OUTPUT/"
                        "trace_view.json"
                    ),
                ),
                patch.object(tool, "_stop_service") as stop_service,
            ):
                trace = tool.run_profile(config)

            self.assertTrue(trace.name == "trace_view.json")
            self.assertTrue(trace.is_file())
            self.assertTrue((trace.parent / "run.json").is_file())
            self.assertTrue((trace.parent / "response.json").is_file())
            self.assertEqual(
                [call[:2] for call in http_calls],
                [
                    ("GET", "/health"),
                    ("POST", "/v1/chat/completions"),
                    ("POST", "/start_profile"),
                    ("POST", "/stop_profile"),
                ],
            )
            chat_batch.assert_called_once()
            stop_service.assert_called_once()

    def test_command_and_http_helpers_surface_errors_and_preserve_payload(self):
        tool = _load_tool()
        failed = CompletedProcess(["bad"], 7, "", "boom")
        with (
            patch.object(tool.subprocess, "run", return_value=failed),
            self.assertRaisesRegex(RuntimeError, "boom"),
        ):
            tool._run(["bad"])

        config = tool.ProfileConfig.from_mapping(
            {
                "container": "qwen3_parallel_nightly",
                "model": "/models/Qwen3-8B",
                "device": "0",
            }
        )
        with patch.object(
            tool,
            "_docker_exec",
            return_value=CompletedProcess([], 0, "ok", ""),
        ) as docker_exec:
            result = tool._container_http(
                config,
                "POST",
                "/v1/chat/completions",
                {"prompt": "真实提示词"},
            )

        self.assertEqual(result.stdout, "ok")
        args, kwargs = docker_exec.call_args
        self.assertEqual(args[0], config.container)
        self.assertIn("/v1/chat/completions", args[1][-1])
        self.assertIn("真实提示词", kwargs["input_text"])

    def test_batch_request_and_runtime_version_helpers(self):
        tool = _load_tool()
        config = tool.ProfileConfig.from_mapping(
            {
                "container": "qwen3_parallel_nightly",
                "model": "/models/Qwen3-8B",
                "device": "0",
                "num_requests": 16,
                "concurrency": 4,
            }
        )
        request = tool.build_chat_request(config, "qwen-profile")
        with patch.object(
            tool,
            "_docker_exec",
            return_value=CompletedProcess([], 0, "{}", ""),
        ) as docker_exec:
            tool._container_chat_batch(config, request)

        batch_call = docker_exec.call_args
        batch_config = json.loads(batch_call.kwargs["input_text"])
        self.assertEqual(batch_config["num_requests"], 16)
        self.assertEqual(batch_config["concurrency"], 4)
        self.assertEqual(batch_config["request"]["model"], "qwen-profile")

        versions_json = json.dumps(
            {
                "vllm": "0.25.1",
                "vllm_ascend": "0.13.0",
                "torch": "2.10.0",
                "torch_npu": "2.10.0.post2",
            }
        )
        with patch.object(
            tool,
            "_docker_exec",
            return_value=CompletedProcess([], 0, versions_json, ""),
        ):
            versions = tool._get_runtime_versions(config.container)
        self.assertEqual(versions["vllm"], "0.25.1")
        self.assertEqual(versions["torch_npu"], "2.10.0.post2")

    def test_health_analysis_cleanup_and_main_helpers(self):
        tool = _load_tool()
        config = tool.ProfileConfig.from_mapping(
            {
                "container": "qwen3_parallel_nightly",
                "model": "/models/Qwen3-8B",
                "device": "0",
            }
        )
        with patch.object(
            tool,
            "_container_http",
            return_value=CompletedProcess([], 0, "", ""),
        ) as health:
            tool._wait_for_health(config, "/workspace/service.log")
        health.assert_called_once_with(config, "GET", "/health", check=False)

        analysis_output = (
            "profiler output\n"
            "TRACE_IN_CONTAINER=/workspace/profile/raw/"
            "rank0_1_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json\n"
        )
        with patch.object(
            tool,
            "_docker_exec",
            return_value=CompletedProcess([], 0, analysis_output, ""),
        ):
            trace = tool._analyse_in_container(
                config.container,
                "/workspace/profile/raw",
            )
        self.assertTrue(trace.endswith("/trace_view.json"))

        with patch.object(tool, "_docker_exec") as docker_exec:
            tool._stop_service(
                config.container,
                "/workspace/profile/service.pid",
                "qwen-profile-run",
            )
        self.assertFalse(docker_exec.call_args.kwargs["check"])

        with self.assertRaisesRegex(ValueError, "Invalid base64"):
            tool._decode_config("not-base64!")
        self.assertEqual(tool.main(["--config-b64", "not-base64!"]), 1)


if __name__ == "__main__":
    unittest.main()
