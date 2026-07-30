# Parallel benchmark platform TDD evidence

## Scope

The test-first increment covers:

- quick and boundary PP/TP/EP matrix expansion;
- reserved CP capability adaptation;
- streaming TTFT, TPOT, ITL, and percentile calculations;
- topology-aware Ascend die selection;
- atomic state persistence and resume semantics;
- kernel, memory, and HCCL classification;
- PP bubble and weighted contiguous stage partition calculations;
- dependency-free comparison charts and Markdown/HTML reports.

## RED

Command:

```text
python tests\ut\examples\test_parallel_bench_tool.py
```

Initial result:

```text
ModuleNotFoundError: No module named 'benchmark_remote'
```

This established that the new test contract did not pass against the previous
single-NPU profiling implementation.

## GREEN

Final local commands:

```text
python tests\ut\examples\test_parallel_bench_tool.py
python tests\ut\examples\test_pipeline_profile_tool.py
python tests\ut\examples\test_parallel_inference.py
python -m py_compile <all profile_tool Python modules>
```

Results:

- new parallel platform contract: 15 tests passed;
- existing profiling tool regression suite: 10 tests passed;
- existing parallel inference tests: passed;
- all new Python modules compile;
- `parallel_bench.ps1` parses successfully and its `-DryRun` path completes.

The local machine has no Ascend runtime. NPU service startup, multi-rank
profiling, and long-running recovery must therefore be accepted by a separate
server smoke test before a full matrix is submitted.
