# Validation Report

**Project:** Tiny Decision Stack  
**Validation date:** 2026-10-04  
**Repository:** `vitorcalvi/tiny-decision-stack`  
**Status:** Experimental / alpha  

This report records the public evidence used to support the repository's software-reliability claims. It is intentionally conservative: integration success is reported as integration success, not as proof of model accuracy.

## Executive summary

A release-gate validation was performed on commit:

```text
97318489d749ad99940a313aed0a0e1be2a95f28
```

On that exact `main` SHA, the permanent GitHub Actions system completed **28/28 jobs successfully** across the normal CI, broad reliability swarm, production-container runtime, real-model integration, and redundant local-integration workflows.

The result supports the following statement:

> The tested commit passed the repository-controlled software, packaging, portability, concurrency, production-container, and real two-model integration gates.

It does **not** support the stronger statement that LFM2.5 + OpenDecider is more accurate than OpenDecider alone, or that the system achieves 95% decision precision. Those claims require representative held-out labelled data.

## Exact public GitHub Actions evidence

| Gate | Run | Result | What it establishes |
|---|---|---|---|
| CI | https://github.com/vitorcalvi/tiny-decision-stack/actions/runs/37220044438 | PASS | Python 3.10/3.11/3.12 deterministic suite, `pip check`, imports, compile, package build |
| Reliability Swarm | https://github.com/vitorcalvi/tiny-decision-stack/actions/runs/37220044433 | PASS | 22 jobs: portability, fuzz/adversarial shards, dependency edges, clean-wheel installs |
| Production Container Runtime | https://github.com/vitorcalvi/tiny-decision-stack/actions/runs/37220044414 | PASS | Compose validation, image build, real boot, health probes, non-root runtime |
| Real Model | https://github.com/vitorcalvi/tiny-decision-stack/actions/runs/37220044417 | PASS | actual LFM2.5 + OpenDecider CPU loading and repeated inference |
| Local Dependency Integration | https://github.com/vitorcalvi/tiny-decision-stack/actions/runs/37220044445 | PASS | redundant full local-extra/model/Compose/image integration path |

All five runs point at the same commit SHA shown above.

## Job topology

### Normal CI: 3 jobs

- Python 3.10
- Python 3.11
- Python 3.12

Each lane installs the development package, checks dependency consistency, runs pytest, compiles source/tests, imports the API, and builds distributable artifacts.

### Broad reliability swarm: 22 jobs

#### Portability: 9 jobs

Cross-product:

```text
ubuntu-latest  × Python 3.10 / 3.11 / 3.12
macos-latest   × Python 3.10 / 3.11 / 3.12
windows-latest × Python 3.10 / 3.11 / 3.12
```

Purpose: expose platform-specific path, subprocess, environment, packaging, and Python-version assumptions.

#### Seeded adversarial/fuzz: 8 jobs

Seeds:

```text
0 1 2 3 4 5 6 7
```

Each shard uses `TDS_SWARM_CASES=400`, producing thousands of generated contract/routing cases in aggregate. The seed is fixed per shard so any failure can be reproduced exactly.

The swarm mutates valid decision outputs into invalid forms and verifies fail-closed behavior, including unknown choices, NaN confidence, missing probability keys, negative values, invalid normalization, and confidence/probability disagreement.

It also tests API concurrency/backpressure and verifies that an HTTP timeout does not prematurely release an inference slot while the underlying model worker is still running.

#### Dependency-edge: 2 jobs

- declared minimum/floor dependency versions.
- latest/current versions allowed by the declared ranges.

Purpose: detect code that accidentally depends on an undeclared newer API or breaks at the upper edge of supported dependency versions.

#### Clean-wheel: 3 jobs

- Python 3.10
- Python 3.11
- Python 3.12

Each lane builds a wheel, creates a clean virtual environment outside the source tree, installs only the wheel and its dependencies, imports the service, and probes health endpoints.

Purpose: prevent a source-tree-only success from being mistaken for a usable package.

### Production container runtime: 1 job

This job validates the actual production image rather than only the Dockerfile syntax:

1. `docker compose config`
2. production image build
3. actual container start
4. liveness/readiness HTTP probes
5. runtime UID verification

The tested image executed as non-root UID `10001`.

### Real-model integration: 1 job

This job installs the real local inference dependency set and executes actual models on CPU.

It checks:

- `accelerate`, `torch`, `transformers`, and `opendecider` imports.
- OpenDecider's `load` entry point.
- lazy LFM2.5 loading.
- lazy OpenDecider loading.
- LFM semantic normalization.
- OpenDecider typed choice execution.
- exact option-key probability coverage.
- probability range and normalization invariants.
- repeated model execution in the same process.

Two bounded scenarios are executed by `scripts/real_model_smoke.py`. The purpose is to prove the real integration path works, not to score semantic correctness.

### Redundant local integration: 1 job

A separate workflow repeats:

- full `[local,dev]` install.
- dependency API checks.
- real two-model CPU smoke.
- Compose validation.
- production image build.

This redundancy reduces the chance that a single workflow's special setup creates a false sense of integration health.

## Safety/reliability invariants covered by source tests

The repository's deterministic and seeded tests explicitly cover these invariants:

### Request boundary

- state must not be blank.
- question must not be blank.
- at least two options are required.
- option labels/descriptions must not be blank.
- confidence threshold must remain inside `[0,1]`.
- oversized states are rejected before inference.
- excessive option counts are rejected before inference.
- Unicode input is accepted.

### Semantic boundary

- the LFM-facing semantic representation has a strict schema.
- unknown semantic fields are forbidden.
- a small set of safe model-shape deviations are canonicalized.
- unsafe/non-string evidence structures are rejected.
- malformed JSON/model commentary that cannot produce a valid schema is rejected.
- caller state is wrapped inside an explicit `<untrusted_state>` sentinel.

### Decision boundary

- returned choice must be one of the request options.
- confidence must be finite.
- confidence must be inside `[0,1]`.
- probability keys must exactly equal option keys.
- each probability must be finite and inside `[0,1]`.
- probabilities must sum to approximately one.
- selected-option probability must agree with reported confidence.
- invalid backend output raises a validation error rather than becoming an autonomous choice.

### Orchestration boundary

- direct path remains direct when appropriate.
- ambiguous/long state can be normalized.
- low confidence can trigger clarification and re-decision.
- continued low confidence produces abstention.
- explicit preprocessing overrides automatic routing.

### Runtime boundary

- only configured inference capacity is admitted.
- excess concurrent callers receive a busy response.
- caller timeout does not silently create hidden extra model capacity.
- model/backend failures map to service-unavailable behavior.
- readiness distinguishes lazy-not-loaded from ready.

## New credibility evidence added after the baseline run

The repository now also contains:

- `TESTING.md` — full methodology and claim-to-evidence matrix.
- `tests/test_contract_validation.py` — explicit audit-friendly contract tests for critical fail-closed invariants.
- `.github/workflows/validation-evidence.yml` — generates a downloadable evidence bundle for each validated commit.

The evidence workflow includes the exact commit SHA, environment metadata, dependency snapshot, collected test inventory, verbose JUnit test results, package-build metadata, Compose validation, and benchmark smoke output.

This report should therefore be read as a historical baseline plus a reproducibility guide. Newer commits should be judged by their own GitHub Actions results and evidence artifacts.

## What remains unproven

The following are deliberately **not** presented as validated facts:

### Model-quality superiority

There is not yet a representative public held-out dataset proving:

```text
LFM2.5 preprocessing + OpenDecider > OpenDecider alone
```

The repository includes the comparison harness so users can test that hypothesis on their own labelled domain data.

### 95% precision

`--target-precision 0.95` is a target-selection mechanism, not a guarantee. A threshold is acceptable only when the held-out dataset empirically meets the target.

### Prompt-injection immunity

The semantic prompt treats caller state as untrusted data and uses delimiters, and contract tests ensure the delimiters are applied. This is defense in depth, not proof against all prompt injections.

### High-stakes autonomy

The project is not validated for medical, legal, financial-suitability, safety-critical, or other high-stakes autonomous decisions. Such use requires domain-specific governance, validation, and human-control requirements beyond this repository.

### Internet-facing security

The service includes bounded request and concurrency controls, but it is not itself an authentication/TLS/distributed-rate-limit gateway.

## Recommended interpretation for developers

A fair public description is:

> Tiny Decision Stack is an experimental local typed-decision reference implementation with fail-closed output validation, confidence gating/abstention, cross-platform GitHub Actions testing, production-container runtime validation, and real LFM2.5 + OpenDecider integration smoke tests.

Avoid claims such as “production-grade AI accuracy,” “95% accurate,” “prompt-injection proof,” or “better than OpenDecider alone” until representative public benchmark evidence exists.

## Independent reproduction

See [`TESTING.md`](TESTING.md) for exact local commands and the detailed test inventory.
