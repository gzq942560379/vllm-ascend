# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install high-level profiling scopes into the benchmark container.

The benchmark launcher runs an image-provided vLLM Ascend checkout.  Uploading
the profile tool does not update that checkout, so this small, validated patch
keeps the runtime instrumentation in sync with the tool without replacing the
image's complete model runner.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


ENV_ENTRY = '''    # Comma-separated high-level model execution scopes emitted
    # while profiling.
    # Values may include model phases, input preparation, sampling, and
    # pipeline-parallel communication boundaries.
    # The default keeps the existing forward scope only. This value is not sensitive.
    "VLLM_ASCEND_PROFILING_SCOPES": lambda: os.getenv(
        "VLLM_ASCEND_PROFILING_SCOPES", "forward"
    ),
'''

SEMANTIC_IMPORT = (
    "from vllm_ascend.profiler.semantic_scopes "
    "import semantic_profile_context\n"
)

SEMANTIC_SCOPES_SOURCE = '''# SPDX-License-Identifier: Apache-2.0
"""Configurable Python-level semantic scopes for profiling."""

from contextlib import nullcontext

from vllm.v1.utils import record_function_or_nullcontext

from vllm_ascend import envs


ENABLED_PROFILE_SCOPES = frozenset(
    scope.strip()
    for scope in envs.VLLM_ASCEND_PROFILING_SCOPES.split(",")
    if scope.strip()
)


def semantic_profile_context(scope: str):
    if scope not in ENABLED_PROFILE_SCOPES:
        return nullcontext()
    return record_function_or_nullcontext(scope)
'''

RUNNER_HELPERS = '''
def _forward_profile_scope(attn_state: AscendAttentionState | None) -> str:
    """Return the high-level semantic scope for one model forward pass."""
    if attn_state == AscendAttentionState.DecodeOnly:
        return "decode"
    if attn_state == AscendAttentionState.SpecDecoding:
        return "spec_decode"
    if attn_state == AscendAttentionState.ChunkedPrefill:
        return "chunked_prefill"
    if attn_state in (
        AscendAttentionState.PrefillNoCache,
        AscendAttentionState.PrefillCacheHit,
    ):
        return "prefill"
    return "unknown_forward_stage"

'''

OLD_FORWARD_CONTEXT = '            record_function_or_nullcontext("forward"),'
NEW_FORWARD_CONTEXT = '''            _forward_profile_context("forward"),
            _forward_profile_context(_forward_profile_scope(self.attn_state)),'''
SEMANTIC_FORWARD_CONTEXT = '''            semantic_profile_context("forward"),
            semantic_profile_context(_forward_profile_scope(self.attn_state)),'''

WORKER_SUBCLASS = '''
class ProfiledAsyncIntermediateTensors(AsyncIntermediateTensors):
    """Expose the lazy pipeline receive wait as a Python trace scope."""

    def wait_for_comm(self) -> None:
        with semantic_profile_context("pp_recv_wait"):
            super().wait_for_comm()


'''

WORKER_EXECUTE_WRAPPER = '''    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        with semantic_profile_context("worker_step"):
            return self._execute_model(scheduler_output)

'''


def patch_envs_source(source: str) -> str:
    if '"VLLM_ASCEND_PROFILING_SCOPES"' in source:
        return source
    marker = "# end-env-vars-definition"
    marker_index = source.find(marker)
    if marker_index < 0:
        raise RuntimeError("envs.py is missing the env-vars definition marker")
    closing_index = source.rfind("}\n", 0, marker_index)
    if closing_index < 0:
        raise RuntimeError("envs.py is missing the env_variables closing brace")
    return source[:closing_index] + ENV_ENTRY + source[closing_index:]


def patch_runner_source(source: str) -> str:
    if SEMANTIC_IMPORT not in source:
        import_anchor = (
            "from vllm_ascend.ascend_config import get_ascend_config\n"
        )
        if import_anchor not in source:
            raise RuntimeError(
                "model_runner_v1.py is missing the Ascend import anchor"
            )
        source = source.replace(
            import_anchor, SEMANTIC_IMPORT + import_anchor, 1
        )

    if "def _forward_profile_scope(" not in source:
        helper_anchor = "class ExecuteModelState(NamedTuple):\n"
        if helper_anchor not in source:
            raise RuntimeError(
                "model_runner_v1.py is missing ExecuteModelState"
            )
        source = source.replace(
            helper_anchor, RUNNER_HELPERS + helper_anchor, 1
        )

    if OLD_FORWARD_CONTEXT in source:
        source = source.replace(
            OLD_FORWARD_CONTEXT, SEMANTIC_FORWARD_CONTEXT, 1
        )
    elif NEW_FORWARD_CONTEXT in source:
        source = source.replace(
            NEW_FORWARD_CONTEXT, SEMANTIC_FORWARD_CONTEXT, 1
        )
    elif SEMANTIC_FORWARD_CONTEXT not in source:
        raise RuntimeError(
            "model_runner_v1.py is missing the model forward scope"
        )
    replacements = {
        'record_function_or_nullcontext("prepare input")':
            'semantic_profile_context("prepare_input")',
        'record_function_or_nullcontext("post process")':
            'semantic_profile_context("post_process")',
        'record_function_or_nullcontext("sample_token")':
            'semantic_profile_context("sample_token")',
        'record_function_or_nullcontext("draft_token")':
            'semantic_profile_context("draft_token")',
        'record_function_or_nullcontext("async_state_update")':
            'semantic_profile_context("async_state_update")',
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    return source


def patch_worker_source(source: str) -> str:
    if SEMANTIC_IMPORT not in source:
        import_anchor = (
            "from vllm_ascend.profiler.torch_npu_profiler "
            "import TorchNPUProfilerWrapper\n"
        )
        if import_anchor not in source:
            raise RuntimeError("worker.py is missing the profiler import")
        source = source.replace(
            import_anchor, SEMANTIC_IMPORT + import_anchor, 1
        )
    if "class ProfiledAsyncIntermediateTensors(" not in source:
        class_anchor = "class NPUWorker(WorkerBase):\n"
        if class_anchor not in source:
            raise RuntimeError("worker.py is missing NPUWorker")
        source = source.replace(
            class_anchor, WORKER_SUBCLASS + class_anchor, 1
        )
    if "return self._execute_model(scheduler_output)" not in source:
        execute_anchor = '''    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
'''
        if execute_anchor not in source:
            raise RuntimeError("worker.py is missing execute_model")
        source = source.replace(
            execute_anchor,
            WORKER_EXECUTE_WRAPPER
            + execute_anchor.replace("def execute_model", "def _execute_model"),
            1,
        )
    source = source.replace(
        '''        if self._pp_send_work:
            for handle in self._pp_send_work:
                handle.wait()
            self._pp_send_work = []''',
        '''        if self._pp_send_work:
            with semantic_profile_context("pp_send_wait"):
                for handle in self._pp_send_work:
                    handle.wait()
            self._pp_send_work = []''',
        1,
    )
    source = source.replace(
        '''            tensor_dict, comm_handles, comm_postprocess = get_pp_group().irecv_tensor_dict(
                all_gather_group=all_gather_group
            )''',
        '''            with semantic_profile_context("pp_recv_submit"):
                tensor_dict, comm_handles, comm_postprocess = get_pp_group().irecv_tensor_dict(
                    all_gather_group=all_gather_group
                )''',
        1,
    )
    source = source.replace(
        "intermediate_tensors = AsyncIntermediateTensors(",
        "intermediate_tensors = ProfiledAsyncIntermediateTensors(",
        1,
    )
    source = source.replace(
        '''        self._pp_send_work = get_pp_group().isend_tensor_dict(
            output.tensors,
            all_gather_group=all_gather_group,
        )''',
        '''        with semantic_profile_context("pp_send_submit"):
            self._pp_send_work = get_pp_group().isend_tensor_dict(
                output.tensors,
                all_gather_group=all_gather_group,
            )''',
        1,
    )
    source = source.replace(
        '''    def sample_tokens(self, grammar_output: "GrammarOutput") -> ModelRunnerOutput | AsyncModelRunnerOutput:
        return self.model_runner.sample_tokens(grammar_output)''',
        '''    def sample_tokens(self, grammar_output: "GrammarOutput") -> ModelRunnerOutput | AsyncModelRunnerOutput:
        with semantic_profile_context("sample_step"):
            return self.model_runner.sample_tokens(grammar_output)''',
        1,
    )
    return source


def _write_validated(path: Path, source: str) -> None:
    compile(source, str(path), "exec")
    temporary = path.with_suffix(path.suffix + ".profile-scopes.tmp")
    temporary.write_text(source, encoding="utf-8")
    temporary.replace(path)


def install(package_root: Path) -> tuple[Path, Path, Path, Path]:
    envs_path = package_root / "envs.py"
    runner_path = package_root / "worker" / "model_runner_v1.py"
    worker_path = package_root / "worker" / "worker.py"
    semantic_path = package_root / "profiler" / "semantic_scopes.py"
    for path in (envs_path, runner_path, worker_path):
        if not path.is_file():
            raise RuntimeError(f"vLLM Ascend source file not found: {path}")

    envs_source = patch_envs_source(envs_path.read_text(encoding="utf-8"))
    runner_source = patch_runner_source(
        runner_path.read_text(encoding="utf-8")
    )
    worker_source = patch_worker_source(
        worker_path.read_text(encoding="utf-8")
    )
    _write_validated(envs_path, envs_source)
    _write_validated(semantic_path, SEMANTIC_SCOPES_SOURCE)
    _write_validated(runner_path, runner_source)
    _write_validated(worker_path, worker_source)
    return envs_path, semantic_path, runner_path, worker_path


def find_package_root() -> Path:
    spec = importlib.util.find_spec("vllm_ascend")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("vllm_ascend package is not installed")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path)
    arguments = parser.parse_args()
    package_root = (
        arguments.package_root.resolve()
        if arguments.package_root
        else find_package_root()
    )
    installed_paths = install(package_root)
    for path in installed_paths:
        print(f"profiling scopes installed: {path}")


if __name__ == "__main__":
    main()
