# Contributing

Thanks for improving Tiny Decision Stack.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
python -m compileall -q src tests
python -m build
```

Use `.[local,dev]` only when testing the real model dependencies.

## Contribution rules

- Keep the architecture small, deterministic where possible, and fail-closed around autonomous decisions.
- Add or update tests for every behavior change.
- Never add model-quality claims without reproducible labelled evaluation.
- Do not commit model weights, credentials, private datasets, or generated caches.
- Preserve abstention semantics: uncertainty and infrastructure failure are different conditions.
- For changes to backend parsing/validation, include adversarial cases (invalid choice, NaN/inf, malformed distributions, schema drift).

## Benchmarks

If a change claims better decision quality, include the dataset hash, model IDs/revisions, hardware, command, precision, coverage, effective accuracy, abstention/clarification rates, and latency. Prefer held-out domain data and compare against the direct OpenDecider-only baseline.
