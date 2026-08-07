# MoE weight-offload benchmark

Use `parallel_bench.ps1` with the supplied smoke configuration:

```powershell
.\parallel_bench.ps1 -Action Submit `
  -ConfigFile .\configs\qwen3_30b_a3b_offload.json
```

The `offload` object maps to vLLM prefetch-offload CLI flags:

```json
"offload": {
  "backend": "prefetch",
  "group_size": 4,
  "num_in_group": 1,
  "prefetch_step": 1,
  "params": ["w13_weight", "w2_weight"]
}
```

`w13_weight` and `w2_weight` select fused MoE expert weights. An empty
`params` list offloads every eligible parameter in the selected layers.

For an HBM A/B comparison, keep `execution.kv_cache_memory_bytes` identical
between the `none` and `prefetch` runs. Otherwise vLLM may use memory released
by offloading to enlarge the KV cache.

Before leasing NPUs, the controller verifies that the container CLI supports
all requested offload flags. After startup it writes
`cases/<case_id>/offload_evidence.json` from the service log and fails the case
if no offloader initialization evidence is found. The generated
`command.json` records the effective offload configuration and CLI flags.
