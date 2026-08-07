# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
    """Record ``scope`` only when explicitly enabled for this process."""
    if scope not in ENABLED_PROFILE_SCOPES:
        return nullcontext()
    return record_function_or_nullcontext(scope)
