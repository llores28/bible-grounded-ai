# Implementation Status

Snapshot date: 2026-08-11

## Implemented and verified locally

- Versioned moral constitution, hermeneutics, theology, safety, data, model, licensing, and risk policies.
- Executable rules for all Ten Commandments, with commandments 5-10 enforced as the interpersonal hard floor.
- Evidence classes and structured `MoralAnswer`, `CommandmentRule`, and `CommandmentAssessment` contracts.
- Prophetic rule registry with deterministic test vectors, unit preservation, assumptions, named schools, and counter-readings.
- Approved-source SQLite passage retrieval and typed canonical graph edges; organizational sources are excluded from biblical retrieval and graph evidence.
- Retrieval-scoped exact quotation and citation verification.
- Pastoral checks for abuse, coercion, self-harm, violence, medical refusal, financial exploitation, scrupulosity, religious paranoia, and false divine-authority claims.
- Local OpenAI-compatible inference orchestration with correction, refusal, escalation, and release-only rendering.
- Review-aware SFT and preference validators, stable answer rendering, pinned Apertus 8B QLoRA/DPO configs, CUDA inspection, run manifests, and non-waivable release gates.
- Public commandment/adversarial cases and sealed-set custody requirements.

## Deliberately not claimed

- No biblical corpus is approved or ingested.
- No expert-reviewed training example is accepted (`0/3,000`).
- No independently reviewed preference pair is accepted (`0/1,000`).
- No sealed acceptance case is in custody (`0/500`).
- No QLoRA or DPO run has executed.
- No trained adapter, benchmark result, scholar approval, or public model release exists.

`python -m biblical_moral_ai validate` verifies implementation integrity. `python -m biblical_moral_ai preflight` must remain blocked until corpus provenance, licensing, hashes, accepted data counts, and review requirements are real. These counters must never be advanced with generated or unreviewed placeholders.

