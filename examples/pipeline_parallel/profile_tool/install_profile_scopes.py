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
    # Valid values are forward, prefill, decode, chunked_prefill, and spec_decode.
    # The default keeps the existing forward scope only. This value is not sensitive.
    "VLLM_ASCEND_PROFILING_SCOPES": lambda: os.getenv(
        "VLLM_ASCEND_PROFILING_SCOPES", "forward"
    ),
'''

RUNNER_IMPORT = "from vllm_ascend import envs as envs_ascend\n"

RUNNER_HELPERS = '''
ENABLED_FORWARD_PROFILE_SCOPES = frozenset(
    scope.strip()
    for scope in envs_ascend.VLLM_ASCEND_PROFILING_SCOPES.split(",")
    if scope.strip()
)


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


def _forward_profile_context(scope: str):
    if scope not in ENABLED_FORWARD_PROFILE_SCOPES:
        return nullcontext()
    return record_function_or_nullcontext(scope)


'''

OLD_FORWARD_CONTEXT = '            record_function_or_nullcontext("forward"),'
NEW_FORWARD_CONTEXT = '''            _forward_profile_context("forward"),
            _forward_profile_context(_forward_profile_scope(self.attn_state)),'''


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
    if "_forward_profile_scope(self.attn_state)" in source:
        return source

    import_anchor = "from vllm_ascend.ascend_config import get_ascend_config\n"
    if import_anchor not in source:
        raise RuntimeError("model_runner_v1.py is missing the Ascend import anchor")
    source = source.replace(
        import_anchor, RUNNER_IMPORT + import_anchor, 1
    )

    helper_anchor = "class ExecuteModelState(NamedTuple):\n"
    if helper_anchor not in source:
        raise RuntimeError("model_runner_v1.py is missing ExecuteModelState")
    source = source.replace(helper_anchor, RUNNER_HELPERS + helper_anchor, 1)

    if source.count(OLD_FORWARD_CONTEXT) != 1:
        raise RuntimeError(
            "model_runner_v1.py does not contain exactly one forward scope"
        )
    return source.replace(OLD_FORWARD_CONTEXT, NEW_FORWARD_CONTEXT, 1)


def _write_validated(path: Path, source: str) -> None:
    compile(source, str(path), "exec")
    temporary = path.with_suffix(path.suffix + ".profile-scopes.tmp")
    temporary.write_text(source, encoding="utf-8")
    temporary.replace(path)


def install(package_root: Path) -> tuple[Path, Path]:
    envs_path = package_root / "envs.py"
    runner_path = package_root / "worker" / "model_runner_v1.py"
    for path in (envs_path, runner_path):
        if not path.is_file():
            raise RuntimeError(f"vLLM Ascend source file not found: {path}")

    envs_source = patch_envs_source(envs_path.read_text(encoding="utf-8"))
    runner_source = patch_runner_source(
        runner_path.read_text(encoding="utf-8")
    )
    _write_validated(envs_path, envs_source)
    _write_validated(runner_path, runner_source)
    return envs_path, runner_path


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
    envs_path, runner_path = install(package_root)
    print(f"profiling scopes installed: {envs_path}")
    print(f"profiling scopes installed: {runner_path}")


if __name__ == "__main__":
    main()
