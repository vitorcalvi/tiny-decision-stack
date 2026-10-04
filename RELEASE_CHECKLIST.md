# Release Checklist

A release is green only when every repo-controlled mandatory gate is green. External/heavy model gates may be yellow for an experimental release but must be disclosed.

## Mandatory green gates

- [ ] CI passes on Python 3.10, 3.11, and 3.12.
- [ ] `pytest` passes with adversarial validation, orchestration, API and benchmark tests.
- [ ] `python -m compileall -q src tests` passes.
- [ ] `python -m build` produces wheel and sdist.
- [ ] Package/API import succeeds in a clean CI environment.
- [ ] Invalid backend outputs fail closed.
- [ ] Request size/option limits and infrastructure failures do not produce autonomous choices.
- [ ] README claims match measured evidence.
- [ ] Example benchmark data is identified as smoke-only.

## Integration gates

- [ ] Scheduled/manual `[local,dev]` workflow resolves/imports Torch, Transformers and OpenDecider APIs.
- [ ] Real LFM + OpenDecider smoke test completed on intended deployment hardware.
- [ ] `docker compose config` validates.
- [ ] Docker image builds and `/health/live` succeeds.
- [ ] After first successful inference, `/health/ready` reports both backends ready.

## Model-quality gate before claiming production accuracy

Run a held-out labelled dataset using `--compare-direct`. Record dataset SHA-256, model IDs/revisions, hardware, dependency versions, latency and memory. Demonstrate that the stack improves the required coverage/precision frontier over direct-only behavior before advertising that improvement.

Never convert a skipped/unavailable heavy test into a green claim; mark it yellow and explain why.
