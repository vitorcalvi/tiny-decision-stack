# Benchmarking

Use your own labelled JSONL decisions to find the confidence threshold that maximizes autonomous coverage at a target precision.

Each line:

```json
{"state":"Customer was charged twice.","question":"What action?","options":{"refund":"Verified duplicate charge","escalate":"Needs human authorization"},"label":"refund","preprocess":"auto"}
```

Run:

```bash
python -m tiny_decision_stack.benchmark decisions.jsonl --target-precision 0.95
```

The command sweeps confidence thresholds and reports:

- `precision`: correctness among non-abstained decisions
- `coverage`: fraction handled autonomously
- `effective_accuracy`: correct autonomous decisions / all cases
- `clarification_rate`: fraction that required the LFM clarification pass
- `sweet_spot`: highest-coverage threshold meeting the requested target precision

For trustworthy production numbers, benchmark on decisions that were not used to tune prompts, schemas, thresholds, or model fine-tuning.
