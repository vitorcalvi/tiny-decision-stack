# Credible benchmark v1

This suite is the repository's model-quality benchmark. It is deliberately separate from `../example-smoke.jsonl`, which is only an integration/demo fixture.

## Frozen evaluation protocol

The suite contains **354 calibration cases** and **600 disjoint holdout cases**:

| Task | Calibration | Holdout | What it probes |
|---|---:|---:|---|
| BoolQ | 100 train | 200 validation | evidence-grounded yes/no decisions |
| ARC-Challenge | 100 train | 200 test | multiple-choice reasoning |
| Banking77 | 154 train | 200 test | 77-way real-world intent routing |

Banking77 calibration uses 2 examples per intent. Banking77 holdout covers all 77 intents with 2 examples per intent plus one extra example from 46 deterministically selected intents.

One global confidence threshold is selected **per variant using calibration only**. The selection rule is: maximize coverage among thresholds whose calibration selective precision reaches the configured target (90% by default). If no candidate reaches the target, choose the highest observed calibration precision, then coverage, then the lower threshold. The 600-case holdout is never used for threshold selection.

Two variants are evaluated on exactly the same cases:

- `full_stack`: production LFM semantic preprocessing/clarification + OpenDecider.
- `direct_only`: OpenDecider on raw state, with semantic preprocessing and clarification disabled.

The evaluator also reports raw top-1 accuracy at threshold `0.0`, so quality remains comparable even if selective precision targets are unreachable.

## Reproduce the frozen data

The materialized JSONL files are committed so evaluation does not depend on upstream availability. To regenerate them from pinned sources:

```bash
uv run --python 3.11 --with 'datasets>=4,<5' \
  python scripts/build_credible_dataset.py
```

Then verify that `calibration.jsonl` and `holdout.jsonl` match the SHA256 values in `manifest.json`.

## Run real-model evaluation

```bash
pip install -e '.[local,dev]'
MODEL_DEVICE=cpu python scripts/run_credible_benchmark.py \
  --benchmark-dir benchmarks/credible-v1 \
  --output-dir benchmark-results/credible-v1 \
  --target-precision 0.90
```

Outputs:

- `report.json`: machine-readable metrics, selected thresholds, confidence intervals, environment and provenance.
- `summary.md`: concise human-readable results.
- `per_case_traces.jsonl`: first/second-stage scores and timing for reproducibility/audit.

The evaluator captures each model stage once per case and simulates the candidate thresholds from those traces. Tests verify that this simulation matches `DecisionOrchestrator` semantics for direct, normalized and clarified paths.

## Statistical reporting

The report includes:

- effective accuracy (`correct autonomous answers / all holdout cases`),
- selective precision (`correct / answered`),
- coverage (`answered / all`),
- 95% Wilson intervals for effective accuracy and selective precision,
- paired bootstrap 95% confidence intervals for full-stack minus direct-only effective accuracy,
- raw top-1 accuracy at threshold 0,
- mean/p50/p95 latency,
- normalization and clarification rates,
- per-task breakdowns.

## Interpretation boundary

This is credible **zero-shot held-out evaluation** because threshold selection and holdout scoring are separated and the dataset files are frozen by hash. It is not contamination-proof: BoolQ, ARC and Banking77 are public benchmarks and could have appeared in model pretraining. Results therefore support claims about performance on this frozen suite, not universal decision quality or a guarantee for a deployment domain.

See `LICENSES.md` and `manifest.json` for source attribution, licenses, revisions, selection seed and exact hashes.
