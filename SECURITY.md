# Security Policy

Tiny Decision Stack is experimental software intended for trusted/local deployment unless protected by an external gateway.

## Supported versions

Security fixes target the latest release on `main`.

## Reporting a vulnerability

Please open a private GitHub security advisory for this repository when possible. Do not include secrets, private datasets, or exploitable production endpoints in a public issue.

## Threat model and known boundaries

- Caller-controlled `state`, `question`, and option descriptions are untrusted input.
- The semantic prompt marks caller state as data and validates the returned decision-state schema, but generative models are not assumed to be prompt-injection-proof.
- Decision backend output is validated before autonomous return; invalid choices/confidences/probabilities fail closed.
- The service does not include authentication, TLS termination, tenant isolation, or distributed rate limiting. Add those controls before internet exposure.
- Inference is resource intensive. Request size, option count, timeout, and concurrency limits reduce accidental/hostile resource exhaustion but are not a full denial-of-service defense.
- Model weights and model-serving dependencies are external supply-chain inputs. Pin model revisions in controlled production deployments and review upstream licenses/advisories.

Do not use this project as the sole control for safety-critical, medical, legal, financial, security-sensitive, or irreversible actions without domain-specific validation and human fallback.
