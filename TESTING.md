# Testing and Reliability Methodology

This document explains exactly what Tiny Decision Stack tests, how the GitHub Actions reliability swarm is structured, which claims the tests support, and which claims they deliberately do **not** support.

The goal is reproducibility and falsifiability: a reader should be able to inspect the test source, rerun the same gates, and distinguish software reliability from model-quality claims.

## Reliability philosophy

Tiny Decision Stack is treated as a bounded decision service. Reliability therefore means more than “the unit tests pass.” The repository tests several independent failure classes:

1. **Input contract failures** — malformed or unsafe requests must be rejected before inference.
2. **Semantic-schema failures** — LFM output must match the strict decision-state schema.
3. **Decision-output failures** — invalid choices, probabilities, or confidence values must fail closed.
4. **Orchestration failures** — direct, normalize, clarify, and abstain paths must behave deterministically.
5. **Concurrency failures** — capacity must not be overcommitted when inference is slow or timed out.
6. **Packaging failures** — a built wheel must install and import in a clean environment.
7. **Dependency failures** — both declared minimum versions and current dependency versions must work.
8. **Portability failures** — the package must pass on Linux, macOS, and Windows across Python 3.10–3.12.
9. **Container failures** — the production image must build, boot, answer health probes, and run non-root.
10. **Real-model integration failures** — the actual LFM2.5 + OpenDecider dependency stack must load and execute on CPU.
11. **Model-quality failures** — these require a separate held-out labelled benchmark and are not inferred from smoke/integration tests.

## Test source inventory

### `tests/test_api.py`

API-level deterministic tests verify:

- `/health` and `/health/live` liveness behavior.
- `/health/ready` backend readiness reporting without forcing model downloads.
- successful `/decide` serialization.
- backend failures become HTTP 503 rather than confident decisions.
- request-state size enforcement.
- maximum-option enforcement.
- Unicode request/response behavior.

These tests use controlled fake backends so API semantics can be tested deterministically without downloading model weights.

### `tests/test_orchestrator.py`

Core decision-flow tests verify:

- confident direct decisions skip semantic preprocessing.
- low-confidence first decisions trigger semantic normalization/clarification and a second decision.
- persistent uncertainty produces abstention (`choice: null`).
- long state is normalized before the first decision.
- short ambiguous prose is normalized under `preprocess=auto`.
- structured JSON remains on the direct path under `preprocess=auto`.
- explicit `direct` and `normalize` modes override automatic routing.
- confidence thresholds at `0.0` and `1.0` behave correctly.
- invalid backend choices fail closed.
- NaN and infinity confidence fail closed.
- confidence outside `[0,1]` fails closed.
- missing probability keys fail closed.
- probability sums outside tolerance fail closed.
- confidence/probability disagreement fails closed.
- safe small-model schema deviations are canonicalized.
- unsafe semantic output shapes are rejected.
- Unicode and prompt-like text are handled as data.
- blank request fields and too-few-options requests are rejected.

### `tests/test_contract_validation.py`

This file intentionally duplicates critical safety contracts with explicit, audit-friendly test names. It verifies:

- blank/whitespace-only state rejection.
- blank/whitespace-only question rejection.
- minimum of two decision options.
- non-empty option keys and descriptions.
- confidence-threshold bounds.
- strict semantic schema (`extra="forbid"`).
- canonicalization of safe semantic-model output variations.
- rejection of unsafe evidence shapes.
- fenced JSON parsing.
- bounded recovery of a JSON object surrounded by model commentary.
- rejection of non-JSON, JSON arrays, and malformed JSON.
- caller text remains inside the `<untrusted_state>` sentinel used by the LFM prompt.
- preservation of a valid scored choice.
- unknown choices fail closed.
- NaN/infinite/out-of-range confidence fails closed.
- missing/extra probability keys fail closed.
- NaN/out-of-range probabilities fail closed.
- non-normalized probability sums fail closed.
- confidence/probability disagreement fails closed.
- malformed integer environment settings fail at configuration time.
- malformed/non-finite floating-point settings fail at configuration time.
- configured numeric bounds are enforced.

The explicit contract suite is valuable for external reviewers because each critical invariant has a plainly named regression test instead of existing only inside randomized fuzzing.

### `tests/test_reliability_swarm.py`

This is the seeded adversarial/fuzz and concurrency suite.

Each fuzz shard receives a deterministic `TDS_SWARM_SEED` and runs hundreds of generated cases. Across eight GitHub Actions shards this exercises thousands of generated inputs while remaining reproducible.

It verifies:

- random valid decisions preserve all output invariants.
- random invalid backend mutations always fail closed.
- randomized structured-vs-ambiguous routing remains deterministic.
- Unicode and multilingual text survive generated-state handling.
- API backpressure rejects overcommit while one inference slot is occupied.
- after an HTTP timeout, the inference slot remains reserved until the underlying worker **actually completes**.

That final invariant is particularly important for local model serving: Python cannot safely terminate an arbitrary running inference thread, so releasing the slot merely because the HTTP caller timed out would permit hidden inference accumulation and eventual RAM/VRAM exhaustion.

### `tests/test_benchmark.py`

Benchmark tests verify the metric implementation itself:

- coverage math.
- precision math.
- effective accuracy math.
- abstention accounting.
- target-precision sweet-spot selection.

These tests establish that the benchmark calculator behaves as designed. They do not establish any particular model-quality result.

## GitHub Actions reliability swarm

The public workflows are intentionally split so one class of test cannot hide failures in another.

### 1. Fast CI — `.github/workflows/ci.yml`

Runs on Python 3.10, 3.11, and 3.12. Each lane performs:

```bash
pip install -e '.[dev]'
python -m pip check
pytest -q
python -m compileall -q src tests
python -c "import tiny_decision_stack.api"
python -m build
```

Purpose: fast regression detection, packaging, imports, syntax, and deterministic tests.

### 2. Reliability Swarm — `.github/workflows/swarm-reliability.yml`

The broad swarm contains 22 independent jobs:

- 9 portability lanes: Linux/macOS/Windows × Python 3.10/3.11/3.12.
- 8 deterministic fuzz shards.
- 2 dependency-edge lanes: declared dependency floors and current/latest allowed dependencies.
- 3 clean-wheel lanes: build a wheel and install it into a fresh virtual environment on Python 3.10/3.11/3.12.

The fuzz jobs use:

```text
TDS_SWARM_SEED = shard number
TDS_SWARM_CASES = 400
```

This makes failures replayable rather than “random CI flakiness.”

### 3. Production container runtime — `.github/workflows/swarm-heavy.yml`

The container lane validates:

- `docker compose config`.
- production image build.
- real container boot.
- liveness/readiness HTTP probes.
- non-root runtime identity.

Purpose: catch Dockerfile/Compose/runtime problems that Python-only tests cannot detect.

### 4. Real-model integration — `.github/workflows/swarm-real-model.yml`

This lane installs `.[local,dev]` and uses the actual upstream model libraries on CPU. It validates:

- Torch/Transformers/OpenDecider dependency resolution.
- the OpenDecider `load` API shape.
- real LFM2.5 model loading.
- real OpenDecider model loading.
- actual LFM normalization followed by actual OpenDecider decision execution.
- returned option keys and probability invariants.
- repeated execution in one process.

The real-model smoke deliberately uses `confidence_threshold=0.0`; it is an integration truth test, not an accuracy benchmark.

### 5. Local dependency integration — `.github/workflows/local-integration.yml`

This redundant integration lane repeats the full local-extra install, real dependency API check, real two-model CPU smoke, Compose validation, and production-image build in one workflow. Its purpose is defense in depth against a green result caused by an isolated workflow-specific assumption.

### 6. Validation evidence — `.github/workflows/validation-evidence.yml`

This workflow creates a downloadable evidence bundle for reviewers. It records:

- exact Git SHA.
- UTC timestamp.
- Python and platform information.
- resolved Python dependency list.
- collected pytest test names.
- verbose pytest execution with JUnit XML.
- package build artifacts metadata.
- Compose validation.
- benchmark smoke output.

The bundle is uploaded as a GitHub Actions artifact tied to the exact commit SHA.


### 7. Credible model-accuracy benchmark — `.github/workflows/credible-benchmark.yml`

This is the statistical model-quality lane. It is intentionally separate from the real-model smoke and deterministic reliability swarm. It evaluates **600 frozen holdout cases** (200 BoolQ, 200 ARC-Challenge, 200 Banking77) after choosing one confidence threshold per variant on a separate 354-case calibration split.

The workflow compares the production full stack against an OpenDecider-only baseline on identical cases and reports raw top-1 accuracy, calibrated effective accuracy, selective precision, coverage, 95% Wilson intervals, paired bootstrap 95% confidence intervals, per-task metrics, normalization/clarification rates, and latency. Low model accuracy does not fail CI; only execution, protocol, or data-integrity failures do.

The benchmark is scientifically stronger than the 3-case smoke fixture because holdout labels never participate in threshold selection. The public datasets can still have pretraining contamination, so results should be described as performance on the frozen `credible-v1` suite, not universal decision accuracy.

## How to reproduce locally

### Fast deterministic suite

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
python -m pip check
pytest -vv
python -m compileall -q src tests
python -m build
```

### One deterministic fuzz shard

```bash
TDS_SWARM_SEED=3 TDS_SWARM_CASES=400 \
pytest -vv tests/test_reliability_swarm.py
```

Change the seed to replay another shard.

### Full real-model CPU smoke

```bash
pip install -e '.[local,dev]'
MODEL_DEVICE=cpu python scripts/real_model_smoke.py
```

This may download upstream model weights.

### Production container

```bash
docker compose config
docker compose build
docker compose up
```

Then verify:

```bash
curl -fsS http://localhost:8080/health/live
curl -fsS http://localhost:8080/health/ready
```

### Credible frozen real-model benchmark

```bash
pip install -e '.[local,dev]'
MODEL_DEVICE=cpu python scripts/run_credible_benchmark.py \
  --benchmark-dir benchmarks/credible-v1 \
  --output-dir benchmark-results/credible-v1 \
  --target-precision 0.90
```

The threshold is chosen using `calibration.jsonl` only. `holdout.jsonl` is then evaluated once at that frozen threshold. Do not retune after looking at holdout results.

### Benchmark your own labelled decisions

```bash
python -m tiny_decision_stack.benchmark your-held-out-decisions.jsonl \
  --compare-direct \
  --target-precision 0.95
```

For a deployment claim, your own held-out data should match the intended domain. The included 3-case smoke fixture remains insufficient model-quality evidence.

## Claim-to-evidence matrix

| Claim | Evidence | Status |
|---|---|---|
| Python package installs/builds | CI + clean-wheel lanes | Verified |
| Python 3.10–3.12 support | CI + portability swarm | Verified |
| Linux/macOS/Windows deterministic behavior | 9 portability lanes | Verified for tested deterministic suite |
| Input validation rejects malformed bounded choices | API/orchestrator/contract tests | Verified |
| Invalid model scores fail closed | deterministic + fuzz contract tests | Verified |
| Low confidence can abstain/escalate | orchestrator tests | Verified |
| API limits and backpressure work | API + concurrency swarm tests | Verified |
| Timeout does not silently free occupied inference capacity | concurrency swarm test | Verified |
| Production image builds and boots | container runtime workflow | Verified |
| Container runs non-root | container runtime workflow | Verified |
| Actual LFM2.5 + OpenDecider execute together | real-model + integration workflows | Verified integration |
| LFM+OpenDecider is more accurate than OpenDecider alone | credible-v1 paired holdout benchmark | Claim only if the executed holdout CI/result supports it |
| A selective precision target is achieved | credible-v1 calibration + untouched holdout | Claim only from holdout result; calibration target alone is not evidence |
| Prompt-injection immunity | cannot be established by current tests | **Not claimed** |
| Internet-facing production security | external gateway/auth/rate limiting required | **Not claimed** |
| Suitability for high-stakes autonomous decisions | domain-specific validation required | **Not claimed** |

## Reading a green build correctly

A green reliability swarm means the repository-controlled software contracts, packaging, portability, concurrency safeguards, container runtime, and real-model integration passed for the tested commit.

It does **not** mean the model is correct about every real-world decision. Decision quality is a statistical property of a model + domain + data distribution. That is why this repository separates software reliability gates from labelled model-quality benchmarking.

## Reporting a failure

When filing an issue, include:

- commit SHA.
- operating system and architecture.
- Python version.
- exact failing command.
- failing seed if the problem came from `test_reliability_swarm.py`.
- sanitized request shape if relevant.
- whether the failure used fake backends, real local models, or Docker.

Do not include API keys, private customer data, or model-provider credentials.
