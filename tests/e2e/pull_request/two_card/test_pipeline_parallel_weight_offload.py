# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.model_utils import check_outputs_equal
from tests.e2e.pull_request.utils import PROMPTS_SHORT

MODEL = "Qwen/Qwen3-0.6B"


def _generate_with_pp(**runner_overrides):
    runner_kwargs = {
        "model_name": MODEL,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 2,
        "max_model_len": 512,
        "gpu_memory_utilization": 0.7,
        "enforce_eager": True,
    }
    runner_kwargs.update(runner_overrides)
    with VllmRunner(**runner_kwargs) as runner:
        return runner.generate_greedy(PROMPTS_SHORT, max_tokens=16)


@wait_until_npu_memory_free(target_free_percentage=0.6)
def test_pp2_prefetch_weight_offload_matches_pp2_baseline():
    """Compare real-weight PP inference with and without prefetch offload."""
    baseline_outputs = _generate_with_pp()
    offload_outputs = _generate_with_pp(
        offload_backend="prefetch",
        offload_group_size=4,
        offload_num_in_group=1,
        offload_prefetch_step=1,
        offload_params={"gate_up_proj", "down_proj"},
    )

    check_outputs_equal(
        outputs_0_lst=offload_outputs,
        outputs_1_lst=baseline_outputs,
        name_0="Qwen3-0.6B-PP2-prefetch-offload",
        name_1="Qwen3-0.6B-PP2-baseline",
    )
