# Benchmarking

The benchmark harness is for **labelled operational decisions**. It measures the accuracy/coverage trade-off created by abstention; it is not a general LLM benchmark.

`example-smoke.jsonl` is intentionally tiny and illustrative. **Do not use its results as evidence of model quality.** Bring your own representative labelled JSONL dataset for release claims.

Each line must contain:

```json
{"state":"...","question":"...","options":{"a":"...","b":"..."},"label":"a"}
```

Run the full stack and an OpenDecider-only baseline on the same cases:

```bash
python -m tiny_decision_stack.benchmark my-decisions.jsonl \
  --compare-direct \
  --target-precision 0.95
```

The `direct_only` arm forces `preprocess=direct` and disables semantic clarification. The `stack` arm uses the normal routing/clarification policy. Both use the same decision backend and confidence thresholds.

Reported metrics:

- **coverage**: fraction of cases answered instead of abstained
- **precision**: correctness among answered cases
- **effective_accuracy**: correct autonomous answers / all cases
- **clarification_rate**: fraction that required the semantic retry path
- **mean_final_confidence**: mean final backend confidence
- **mean_latency_ms**: end-to-end process latency for each case
- **dataset_sha256**: hash for reproducibility

For publishable results, also record hardware/OS, Python version, package lock or versions, model IDs/revisions, dataset provenance/size/hash, warmup procedure, memory usage, and the complete command line. Do not claim a target precision such as 95% unless the labelled evaluation actually achieves it on a representative held-out dataset.
