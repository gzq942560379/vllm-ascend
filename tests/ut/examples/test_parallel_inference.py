# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "examples" / "pipeline_parallel" / "parallel_inference.py"
SPEC = importlib.util.spec_from_file_location("parallel_inference", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
parallel_inference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parallel_inference)

ServePlan = parallel_inference.ServePlan
build_environment = parallel_inference.build_environment
build_vllm_command = parallel_inference.build_vllm_command
detect_model_kind = parallel_inference.detect_model_kind
validate_plan = parallel_inference.validate_plan


def _write_config(tmp_path: Path, **overrides) -> Path:
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "num_hidden_layers": 36,
    }
    config.update(overrides)
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return model_path


def test_detects_dense_qwen_model(tmp_path):
    model_path = _write_config(tmp_path)

    assert detect_model_kind(str(model_path)) == "dense"


@pytest.mark.parametrize("expert_key", ["num_experts", "num_local_experts", "n_routed_experts"])
def test_detects_moe_model_from_expert_count(tmp_path, expert_key):
    model_path = _write_config(tmp_path, **{expert_key: 128})

    assert detect_model_kind(str(model_path)) == "moe"


def test_builds_qwen8b_pp_tp_command(tmp_path):
    model_path = _write_config(tmp_path)
    plan = ServePlan(
        model=str(model_path),
        devices=("0", "1", "2", "3"),
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        served_model_name="qwen8b",
        max_model_len=4096,
        max_num_seqs=16,
    )

    command = build_vllm_command(plan)

    assert command[:3] == ["vllm", "serve", str(model_path)]
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert command[command.index("--pipeline-parallel-size") + 1] == "2"
    assert command[command.index("--served-model-name") + 1] == "qwen8b"
    assert "--enable-expert-parallel" not in command


def test_fixed_kv_cache_memory_replaces_gpu_utilization(tmp_path):
    model_path = _write_config(tmp_path)
    plan = ServePlan(
        model=str(model_path),
        devices=("0", "1"),
        pipeline_parallel_size=2,
        kv_cache_memory_bytes=16 * 1024**3,
    )

    command = build_vllm_command(plan)

    assert command[command.index("--kv-cache-memory-bytes") + 1] == "17179869184"
    assert "--gpu-memory-utilization" not in command


def test_rejects_negative_kv_cache_memory(tmp_path):
    model_path = _write_config(tmp_path)
    plan = ServePlan(
        model=str(model_path),
        devices=("0",),
        kv_cache_memory_bytes=-1,
    )

    with pytest.raises(ValueError, match="kv_cache_memory_bytes"):
        validate_plan(plan)


def test_rejects_device_count_that_does_not_match_2d_world_size(tmp_path):
    model_path = _write_config(tmp_path)
    plan = ServePlan(
        model=str(model_path),
        devices=("0", "1", "2"),
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
    )

    with pytest.raises(ValueError, match=r"PP\(2\) × TP\(2\) = 4"):
        validate_plan(plan)


def test_rejects_ep_for_dense_qwen8b(tmp_path):
    model_path = _write_config(tmp_path)
    plan = ServePlan(
        model=str(model_path),
        devices=("0", "1", "2", "3"),
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        enable_expert_parallel=True,
    )

    with pytest.raises(ValueError, match="Dense"):
        validate_plan(plan)


def test_builds_pp_ep_command_for_moe_model(tmp_path):
    model_path = _write_config(
        tmp_path,
        architectures=["Qwen3MoeForCausalLM"],
        num_experts=128,
    )
    plan = ServePlan(
        model=str(model_path),
        devices=("0", "1", "2", "3"),
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        enable_expert_parallel=True,
    )

    command = build_vllm_command(plan)

    assert "--enable-expert-parallel" in command


def test_builds_pp_prefetch_offload_command(tmp_path):
    model_path = _write_config(tmp_path)
    plan = ServePlan(
        model=str(model_path),
        devices=("0", "1"),
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
        offload_backend="prefetch",
        offload_group_size=4,
        offload_num_in_group=1,
        offload_prefetch_step=1,
        offload_params=("gate_up_proj", "down_proj"),
    )

    command = build_vllm_command(plan)

    assert command[command.index("--offload-backend") + 1] == "prefetch"
    assert command[command.index("--offload-group-size") + 1] == "4"
    assert command[command.index("--offload-num-in-group") + 1] == "1"
    assert command[command.index("--offload-prefetch-step") + 1] == "1"
    params_index = command.index("--offload-params")
    assert command[params_index + 1 : params_index + 3] == ["gate_up_proj", "down_proj"]


def test_rejects_invalid_prefetch_group(tmp_path):
    model_path = _write_config(tmp_path)
    plan = ServePlan(
        model=str(model_path),
        devices=("0", "1"),
        pipeline_parallel_size=2,
        offload_backend="prefetch",
        offload_group_size=2,
        offload_num_in_group=3,
    )

    with pytest.raises(ValueError, match="offload_num_in_group"):
        validate_plan(plan)


def test_ascend_uva_fallback_requires_eager_mode(tmp_path):
    model_path = _write_config(tmp_path)
    plan = ServePlan(
        model=str(model_path),
        devices=("0", "1"),
        pipeline_parallel_size=2,
        offload_backend="uva",
        cpu_offload_gb=2,
    )

    with pytest.raises(ValueError, match="enforce-eager"):
        validate_plan(plan)


def test_builds_uva_fallback_environment(tmp_path):
    model_path = _write_config(tmp_path)
    plan = ServePlan(
        model=str(model_path),
        devices=("4", "5"),
        pipeline_parallel_size=2,
        offload_backend="uva",
        cpu_offload_gb=2,
        enforce_eager=True,
    )

    environment = build_environment(plan, {"PATH": "/usr/bin"})

    assert environment["ASCEND_RT_VISIBLE_DEVICES"] == "4,5"
    assert environment["VLLM_WEIGHT_OFFLOADING_DISABLE_UVA"] == "1"
    assert environment["PATH"] == "/usr/bin"
