# Implementation Status

Snapshot date: 2026-08-11

## Implemented and verified locally

- Versioned moral constitution, hermeneutics, theology, safety, data, model, licensing, and risk policies.
- Executable rules for all Ten Commandments, with commandments 5-10 enforced as the interpersonal hard floor.
- Evidence classes and structured `MoralAnswer`, `CommandmentRule`, and `CommandmentAssessment` contracts.
- Prophetic rule registry with deterministic test vectors, unit preservation, assumptions, named schools, and counter-readings.
- Approved-source SQLite passage retrieval and typed canonical graph edges; organizational sources are excluded from biblical retrieval and graph evidence.
- SHA-256 locked importers for KJV, OSHB Hebrew/Aramaic, SBLGNT, and MorphGNT, plus separate Open Scriptures Hebrew/Aramaic and Abbott-Smith Koine Greek lexicon ingestion.
- Atomic evidence-store construction with 66-book/39-book/27-book cardinality checks, safe archive extraction, and an exact-Scripture citation export that excludes dictionary definitions.
- Retrieval-scoped exact quotation and citation verification.
- Pastoral checks for abuse, coercion, self-harm, violence, medical refusal, financial exploitation, scrupulosity, religious paranoia, and false divine-authority claims.
- Local OpenAI-compatible inference orchestration with correction, refusal, escalation, and release-only rendering.
- Review-aware SFT and preference validators, stable answer rendering, pinned Apertus 8B QLoRA/DPO configs, CUDA inspection, run manifests, and non-waivable release gates.
- Public commandment/adversarial cases and sealed-set custody requirements.
- Separate fail-closed pilot workflow targeting 50 SFT records, 20 preference pairs, and 25 evaluation cases; sensitive categories require two registered independent reviewers.
- Curated `draft_only` queue with exactly 50/20/25 unique scenarios, all resolved to approved evidence, plus authoring packets, candidate validation, blinded packet-bound reviews, revision/adjudication rules, finalization, and hash-bound CPU receipts.
- Deterministic AI-authored candidate starting points for all 95 scenarios pass CPU policy and citation checks but count as zero until real independent human review.

## Deliberately not claimed

- Approved third-party sources are not vendored; each operator must reproduce the evidence store from the pinned downloads and hashes.
- No registered independent reviewer exists yet (`0/2` minimum for the pilot).
- No accepted pilot record exists (`0/50` SFT, `0/20` preference, `0/25` evaluation).
- No expert-reviewed training example is accepted (`0/3,000`).
- No independently reviewed preference pair is accepted (`0/1,000`).
- No sealed acceptance case is in custody (`0/500`).
- No QLoRA or DPO run has executed.
- No trained adapter, benchmark result, scholar approval, or public model release exists.

`python -m biblical_moral_ai validate` verifies implementation integrity. `build-evidence --fetch` reproduces the approved evidence database. `pilot-preflight` and production `preflight` remain blocked until their respective reviewed data, reviewer, corpus, and digest requirements are real. These counters must never be advanced with generated or unreviewed placeholders.

