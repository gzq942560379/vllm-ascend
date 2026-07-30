# Request profiling tool TDD evidence

## Source and user journey

No source plan was provided. The journey was derived from the profiling session:
as a Windows user, run one command with a few server/model/version/concurrency
parameters and receive an analysed Ascend `trace_view.json` under a D-drive
directory.

## RED evidence

Command:

```text
python tests/ut/examples/test_pipeline_profile_tool.py
```

Before the implementation existed, all three tests failed with
`FileNotFoundError` for `profile_tool/profile_remote.py`. This was the expected
RED state for the missing tool. The repository pytest bootstrap was not counted
as RED because it failed earlier due to the local machine not having upstream
`vllm` installed.

## Test specification

| # | Guarantee | Test | Type |
|---|---|---|---|
| 1 | Unsafe container names, multi-device values, and invalid ports are rejected before Docker execution | `test_profile_config_rejects_unsafe_or_invalid_values` | Unit |
| 2 | The generated service is PP=1, TP=1, eager, one-sequence, fixed-KV, and enables shape/memory Torch NPU profiling | `test_build_vllm_command_enables_single_request_npu_profiling` | Unit |
| 3 | A run exports exactly one analysed `trace_view.json` | `test_find_trace_view_requires_exactly_one_result` | Unit |
| 4 | UTF-8 prompts survive the Windows-to-server Base64 JSON handoff | `test_decode_config_preserves_a_real_unicode_prompt` | Unit |
| 5 | The full orchestration reaches warm-up, start, measured request, stop, analysis, export, and cleanup | `test_run_profile_executes_and_cleans_up_the_full_control_flow` | Integration with mocked Docker |
| 6 | Subprocess/HTTP failures are surfaced and request payloads are preserved | `test_command_and_http_helpers_surface_errors_and_preserve_payload` | Unit |
| 7 | Health, analysis marker, safe cleanup, and invalid CLI configuration paths behave as expected | `test_health_analysis_cleanup_and_main_helpers` | Unit |
| 8 | ACLGraph omits `--enforce-eager`, while concurrency controls `--max-num-seqs` | `test_aclgraph_and_multi_request_settings_change_the_serve_plan` | Unit |
| 9 | Expected vLLM versions allow build suffixes and reject real mismatches | `test_expected_vllm_version_accepts_build_suffix_and_rejects_mismatch` | Unit |
| 10 | Multi-request count/concurrency and runtime package versions survive the container handoff | `test_batch_request_and_runtime_version_helpers` | Unit |

## GREEN and coverage evidence

Commands:

```text
python tests/ut/examples/test_pipeline_profile_tool.py
python -m trace --count --missing --summary --coverdir <temp> \
  --module unittest tests.ut.examples.test_pipeline_profile_tool
```

Result:

```text
Ran 10 tests
OK
412 lines, 90% profile_remote.py line coverage
```

PowerShell syntax and UTF-8 configuration were also checked with:

```text
profile_trace.ps1 -Server 192.0.2.10 -Prompt "测试中文提示词" -DryRun
```

This completed without making an SSH connection and preserved the Chinese
prompt in the generated configuration.

## Known validation boundary

The Windows machine has no Ascend NPU and no upstream `vllm` installation.
Docker/NPU end-to-end execution therefore requires the user’s Ascend server.
The local validation covers configuration, command generation, trace selection,
mocked end-to-end orchestration, vLLM version validation, eager/ACLGraph
selection, concurrent request configuration, PowerShell parsing, dry-run
behavior, and 90% line coverage of the remote helper.
