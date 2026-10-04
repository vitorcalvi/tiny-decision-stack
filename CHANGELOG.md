# Changelog

## 0.1.1 - Unreleased

Release-hardening pass:

- fail-closed validation for backend choices, confidence and probability distributions
- strict semantic decision-state schema validation
- stronger untrusted-input prompt framing
- liveness/readiness endpoints with honest lazy-loading state
- request-size, option-count, inference timeout and concurrency controls
- safer timed-out inference capacity handling
- deterministic ambiguity-aware auto preprocessing
- adversarial orchestration/API/benchmark tests
- Python 3.10/3.11/3.12 CI matrix and package-build gate
- scheduled/manual real dependency API integration workflow
- reproducible direct-only vs full-stack benchmark comparison mode
- non-root Docker runtime and health checks
- explicit experimental positioning, security policy and release checklist

## 0.1.0

Initial experimental MVP with FastAPI, LFM2.5 semantic preprocessing, OpenDecider-nano decisions, confidence-gated clarification/abstention, Docker packaging, unit tests, and labelled benchmark harness.
