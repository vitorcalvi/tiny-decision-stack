# Tiny Decision Stack

A small local decision system that combines two complementary models:

- **LiquidAI/LFM2.5-350M** — semantic normalization, evidence extraction, constraint preservation, and clarification.
- **manjunathshiva/opendecider-nano** — calibrated final choice with a probability for every option.

The design is intentionally asymmetric: **LFM understands; OpenDecider chooses.**

## Architecture

```text
raw input
   |
   +--> simple/structured ----------------------+
   |                                             |
   +--> LFM2.5-350M                              |
        normalize facts / constraints / risks   |
                    |                            |
                    +------------+---------------+
                                 v
                         OpenDecider-nano
                      choice + probabilities
                                 |
                    +------------+-------------+
                    |                          |
              confidence >= T            confidence < T
                    |                          |
                 return                LFM clarification
                                               |
                                        OpenDecider again
                                               |
                                  confident -> return
                                  uncertain -> abstain
```

This is **not** an ensemble vote. LFM2.5 is the semantic processor and OpenDecider is the policy/decision head.

## Why this pairing?

OpenDecider-nano is optimized for typed decisions (`choice`, `score`, `yes/no`) and returns calibrated option probabilities without free-text generation. LFM2.5-350M is a tiny generative model that can turn messy or long natural language into a compact structured decision state. Together they cover a larger operational decision surface than either model alone while staying small enough for local/edge deployment.

Good fits: agent routing, support triage, tool selection, workflow automation, policy gates, alerts, CRM decisions, and other bounded operational choices.

Not a replacement for a large reasoning model on hard mathematics, deep coding, obscure knowledge, or open-ended strategy. In those cases this project should **abstain**, not invent certainty.

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

Example response:

```json
{
  "choice": "refund_now",
  "confidence": 0.87,
  "probabilities": {
    "refund_now": 0.87,
    "escalate": 0.08,
    "request_information": 0.04,
    "reject": 0.01
  },
  "abstained": false,
  "mode": "normalized",
  "attempts": 1
}
```

## Four execution modes

1. **Direct** — simple structured state goes directly to OpenDecider.
2. **Normalize** — messy/long state goes through LFM2.5, then OpenDecider.
3. **Clarify** — if confidence is low, LFM restructures ambiguity/evidence and OpenDecider runs again.
4. **Abstain** — if confidence is still below your threshold, no autonomous action is selected.

## Install

Python 3.10+.

```bash
git clone https://github.com/vitorcalvi/tiny-decision-stack
cd tiny-decision-stack
python -m venv .venv
source .venv/bin/activate
pip install -e '.[local,dev]'
uvicorn tiny_decision_stack.api:app --host 0.0.0.0 --port 8080
```

The first real request downloads the configured model weights from Hugging Face.

## Configuration

Copy `.env.example` or export variables directly.

| Variable | Default | Meaning |
|---|---|---|
| `LFM_MODEL` | `LiquidAI/LFM2.5-350M` | semantic model |
| `OPENDECIDER_MODEL` | `manjunathshiva/opendecider-nano` | decision model |
| `MODEL_DEVICE` | auto | optional device override |
| `DIRECT_MAX_CHARS` | `500` | auto-mode direct-routing cutoff |
| `DEFAULT_CONFIDENCE_THRESHOLD` | `0.75` | abstention threshold |

## Decision-state schema

LFM is asked to preserve decision-relevant evidence rather than write a normal summary:

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

This structured state is what OpenDecider sees.

## Development

```bash
pip install -e '.[dev]'
pytest -q
```

The unit tests use fake backends and do **not** download model weights.

## Model licenses

This repository contains orchestration code only; it does not redistribute model weights. LFM2.5 and OpenDecider remain subject to their respective upstream model/code licenses. Check those licenses before redistribution or commercial deployment.

## Status

MVP. The core goal is measurable **coverage at a target precision**, not benchmark-chasing. A useful production metric is: _maximize autonomous coverage while maintaining >=95% precision on your own labelled decisions_.
