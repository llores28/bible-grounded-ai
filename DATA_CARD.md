# Data Card

Version: 0.1  
Status: schema and gates implemented; reviewed training corpus not yet collected

## Dataset purpose

The dataset teaches an auditable procedure for Bible-grounded moral reasoning. It is not a corpus of predetermined denominational answers. It trains the model to retrieve and classify evidence, preserve context and source-language limits, assess commandments, state uncertainty, represent serious alternatives, and choose safe practical options.

## Planned composition

| Split | Minimum | Review requirement | Purpose |
| --- | ---: | --- | --- |
| SFT train | 3,000 | Expert-reviewed | Evidence procedure and answer structure |
| Preference train | 1,000 pairs | Independent pair review | DPO preference learning |
| Public development eval | 120 | Reviewed, non-sealed | Engineering regression tests |
| Sealed acceptance eval | 500 | Dual review for high-impact cases | Release decision only |

The machine-readable status is in `data/registry/dataset_manifest.json`. Counts represent accepted, deduplicated records only. Seed outlines and unreviewed drafts never count toward these targets.

The smoke-test pilot is separately scoped to 50 SFT, 20 preference, and 25 evaluation records. Its committed `draft_only` queue is an authoring plan and counts as zero until complete candidates pass deterministic validation and real independent review under `docs/PILOT_REVIEW_WORKFLOW.md`.

## SFT record contract

Every SFT record contains:

- stable record and scenario IDs;
- user request and bounded known/missing facts;
- relevant commandments, including explicit assessments for 5-10;
- source passages, exact approved quotations, immediate context, and canonical links;
- source-language notes and textual witness;
- evidence class for each claim;
- assumptions and strongest counter-reading;
- affected people, power differences, harms, and legal/professional limits;
- conclusion, confidence, alternatives, practical options, and referral;
- optional organizational alignment isolated at zero evidence weight;
- source provenance, license status, reviewer IDs, decisions, and adjudication.

Preference records contain a shared prompt, chosen and rejected structured answers, independent reasons for the preference, hard-floor checks, citation verification, and disagreement/adjudication metadata.

## Collection and review

1. A case author drafts a scenario without target denominational labels.
2. A textual reviewer verifies references, quotations, context, and language notes.
3. A moral/safety reviewer evaluates commandment compliance, affected people, and harms.
4. A second reviewer independently reviews disputed doctrine and every high-impact case.
5. An adjudicator resolves material conflicts and records the rationale.
6. Automated validators check schema, duplicates, citation corpus, source approval, contamination, and split isolation.
7. Only accepted records enter countable manifests.

Review decisions bind cryptographically to a complete candidate packet. Any candidate change invalidates prior decisions. A disagreement can require revision or rejection but cannot be used to accept the disputed version; revised candidates receive fresh blinded reviews.

## Sources and licensing

No corpus may be ingested until its exact revision, retrieval URL, SHA-256 digest, license, attribution text, allowed uses, and legal approval are recorded in `configs/data/source_registry.json`. Public-domain status is jurisdiction-sensitive for the KJV. Morphology and base text may have different licenses and must be tracked separately.

Copyrighted commentaries, lexicons, sermons, denominational books, and proprietary Bible translations are excluded unless explicit training and redistribution rights are documented. Links and bibliographic facts do not imply ingestion permission.

## Quality and leakage controls

- Split by scenario family, passage cluster, and derivation lineage, not only by row.
- Keep paraphrases, preference variants, and source templates in the same split.
- Hash normalized prompts, answers, citations, and source excerpts for near-duplicate review.
- Store sealed evaluation content outside the training repository; commit only its manifest and digest.
- Prohibit test-driven editing against sealed answers.
- Audit organizational names and source IDs to prevent organizational-source leakage into biblical evidence.
- Publish rejected-case categories and reviewer agreement without publishing sensitive personal scenarios.

## Privacy and sensitive content

Use synthetic or consented scenarios. Remove personal identifiers, private pastoral records, credentials, medical records, and identifying abuse details. Never use private communications as training data without documented authorization and minimization review.

## Known limitations

The target counts are design minima, not evidence of doctrinal completeness. The present repository contains no accepted expert-reviewed training records. English-first review and a Protestant canon limit generalization. Historical and linguistic claims remain dependent on the approved source editions and reviewer expertise.

