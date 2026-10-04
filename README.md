# Tiny Decision Stack

> **Status: Experimental v0.1.x** — the orchestration, validation, API, packaging, and fake-backend release gates are tested. Real-model decision quality must be measured on your own labelled decisions before production use.

Tiny Decision Stack is a small local decision service that combines two complementary models:

- **LiquidAI/LFM2.5-350M** — semantic normalization/clarification of messy decision state.
- **manjunathshiva/opendecider-nano** — typed final choice with per-option probabilities.

The design is asymmetric: **LFM structures evidence; OpenDecider chooses.** This repository does not claim that the combined stack is more accurate than OpenDecider alone until a representative labelled comparison proves it.

## Architecture

```text
raw state
  |
  +-- direct structured/clear state ----------------------+
  |                                                       |
  +-- ambiguous/long state -> LFM decision-state schema --+
                                                          v
                                                   OpenDecider
                                                          |
                                              confidence >= threshold
                                               /                   \
                                           return            LFM clarify
                                                               |
                                                         OpenDecider again
                                                           /          \
                                                       return       abstain
```

The semantic schema is validated strictly:

```json
{
  "facts": [],
  "constraints": [],
  "risks": [],
  "evidence_for": {},
  "evidence_against": {},
  "missing_information": []
}
```

Backend decisions are fail-closed: the selected choice must exist in the request options; confidence and probabilities must be finite and in `[0,1]`; probability keys must exactly match the options; probabilities must sum to approximately 1; and selected-option probability must agree with reported confidence. Invalid backend output is rejected instead of being executed autonomously.

## Good fits

Agent routing, support triage, tool selection, workflow automation, policy gates, alerts, CRM decisions, and other **bounded typed choices** with labelled historical examples.

This is not a substitute for a large reasoning model for hard mathematics, deep coding, obscure knowledge, or open-ended strategy. It should abstain or escalate when confidence is insufficient.

## Install

Python 3.10–3.12 are covered by the fast CI matrix.

```bash
git clone https://github.com/vitorcalvi/tiny-decision-stack
cd tiny-decision-stack
python -m venv .venv
source .venv/bin/activate
pip install -e '.[local,dev]'
uvicorn tiny_decision_stack.api:app --host 0.0.0.0 --port 8080
```

Model weights are loaded lazily on first use and may be downloaded by the upstream libraries.

## API

```bash
curl -X POST http://localhost:8080/decide \
  -H 'content-type: application/json' \
  -d '{
    "state": "Customer says they were charged twice and needs it fixed today.",
    "question": "What action should we take?",
    "options": {
      "refund_now": "Verified duplicate payment and refund is permitted",
      "request_information": "Required information is missing",
      "escalate": "Human authorization or exception is required",
      "reject": "Refund conditions are not satisfied"
    },
    "confidence_threshold": 0.75,
    "preprocess": "auto"
  }'
```

Uncertainty produces an abstained response (`choice: null`) after the configured clarification path. Infrastructure/model failures are returned as HTTP 503 errors; malformed requests and configured request-limit violations return 422-class responses. A backend failure is never converted into a confident autonomous choice.

## Health

- `GET /health` and `GET /health/live`: process liveness only.
- `GET /health/ready`: lazy backend state (`not_loaded`, `ready`, or `error`). It deliberately does **not** force model downloads just because a readiness probe ran.

Immediately after startup, readiness can legitimately report `not_ready` until both lazy backends have successfully loaded.

## Auto preprocessing

`preprocess=direct` and `preprocess=normalize` are explicit overrides. `auto` uses a cheap deterministic policy: long text and short prose with ambiguity/conflict markers receive semantic normalization, while valid JSON and compact key/value state can go directly to the decision head. `DIRECT_MAX_CHARS` remains the main size cutoff.

This routing policy is intentionally simple and explainable, not a learned router.

## Configuration

| Variable | Default | Meaning |
|---|---:|---|
| `LFM_MODEL` | `LiquidAI/LFM2.5-350M` | semantic model |
| `OPENDECIDER_MODEL` | `manjunathshiva/opendecider-nano` | decision model |
| `MODEL_DEVICE` | auto | optional device override |
| `DIRECT_MAX_CHARS` | `500` | long-state normalization cutoff |
| `DEFAULT_CONFIDENCE_THRESHOLD` | `0.75` | abstention threshold |
| `MAX_STATE_CHARS` | `20000` | pre-inference request-state limit |
| `MAX_OPTIONS` | `32` | maximum typed options |
| `MAX_CONCURRENT_INFERENCE` | `1` | process-level inference slots |
| `INFERENCE_TIMEOUT_SECONDS` | `120` | HTTP wait timeout |

A timed-out model call may continue running because Python cannot safely kill arbitrary in-process inference. Its capacity slot remains reserved until the worker really finishes, preventing timeout-driven overcommit.

## Benchmark the claim that matters

The meaningful product question is whether semantic preprocessing improves the **coverage-at-target-precision frontier** compared with OpenDecider alone.

Use your own held-out labelled decisions:

```bash
python -m tiny_decision_stack.benchmark my-decisions.jsonl \
  --compare-direct \
  --target-precision 0.95
```

The output reports dataset SHA-256, precision, coverage, effective accuracy, abstention, clarification rate, confidence, and mean latency. `benchmarks/example-smoke.jsonl` is only a smoke/demo fixture and is **not evidence of model quality**. See `benchmarks/README.md` for reproducibility requirements.

## Development and release gates

```bash
pip install -e '.[dev]'
pytest
python -m compileall -q src tests
python -m build
```

Fast CI runs these gates on Python 3.10, 3.11, and 3.12 without downloading model weights. A separate scheduled/manual workflow installs `[local,dev]` and validates the real Torch/Transformers/OpenDecider dependency APIs. That workflow verifies dependency integration, not decision quality.

## Docker

```bash
docker compose up --build
```

The image runs as a non-root user and includes a liveness health check. Hugging Face cache data is persisted in a named volume. Start with `MAX_CONCURRENT_INFERENCE=1`; raise it only after measuring RAM/VRAM on your hardware. The repository does not assume a specific GPU platform.

## Security and production caveats

This project is a local/edge reference service, not an internet-facing gateway. It has request-size and concurrency controls but no built-in authentication or distributed rate limiting. Put authentication, TLS, network policy, and external rate limiting in front of it if exposed beyond a trusted network.

LFM prompts explicitly mark caller state as untrusted data and tell the model to ignore instructions embedded inside it. That reduces a prompt-injection failure mode but is not a proof of prompt-injection immunity. Evaluate adversarial inputs on your own domain before allowing autonomous actions.

See `SECURITY.md` and `RELEASE_CHECKLIST.md`.

## Licensing

This repository's orchestration code is MIT licensed. Model weights remain under their upstream licenses.

- `LiquidAI/LFM2.5-350M` is published with the **LFM Open License v1.0** (`lfm1.0`). Liquid AI states that commercial use under that open license applies to smaller companies below its stated revenue threshold; organizations outside those terms should consult the current upstream license directly.
- OpenDecider code and weights are published as **Apache-2.0**; its `opendecider-nano` base model is identified upstream as MIT licensed.

Always review the current upstream license/NOTICE files before redistribution or commercial deployment; this README is not legal advice.

## What is verified vs not yet claimed

**Verified by repository-controlled gates:** request/schema validation, decision-output validation, direct/normalize/clarify/abstain orchestration, API error semantics, benchmark metric math, Python package build/import, and multi-version fast CI.

**Not established by this repository alone:** that LFM + OpenDecider beats OpenDecider-only on your domain; that a 95% precision target is achieved; production-scale latency/throughput; prompt-injection immunity; or suitability for high-stakes autonomous actions.

## Contributing

See `CONTRIBUTING.md`. Release history is in `CHANGELOG.md`.
