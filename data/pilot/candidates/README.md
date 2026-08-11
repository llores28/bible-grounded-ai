# Pilot candidate authoring

This directory holds fully authored **candidate** records. A candidate is not accepted training data and cannot advance a manifest count.

Create exactly these files:

- `sft.jsonl` with 50 candidate envelopes.
- `preferences.jsonl` with 20 candidate envelopes.
- `evals.jsonl` with 25 candidate envelopes.

Each line follows `schemas/pilot-candidate-envelope.schema.json`:

```json
{"item_id":"SFT-DRAFT-001","candidate_revision":1,"record":{"record_id":"SFT-PILOT-001","scenario_id":"TRUTH-CITATION","category":"truthfulness","status":"candidate","high_impact":false,"answer":{},"provenance":{"author_id":"AUTHOR-ID","created_at":"2026-08-11T00:00:00Z","source_ids":["KJV_EBIBLE_1769"],"license_check_ids":{"KJV_EBIBLE_1769":"LIC-20260811-KJV-EBIBLE"}},"reviews":[]}}
```

The abbreviated `answer` above is illustrative and will fail validation. Authors must provide a complete structured answer. Candidate prompts, categories, source IDs, and references must match `configs/pilot/draft_scenarios.json`; quotations must come from generated authoring packets, never memory.

The committed files in this directory are deterministic AI-authored starting points labeled with `author_id=AUTOMATED-DRAFT-AUTHOR`. They pass CPU checks but have not been human-approved. Human authors/reviewers must inspect and revise them; do not treat generation as review.

The controlled flow is:

1. Run `build-evidence --fetch` and `audit-pilot-drafts`.
2. Run `build-authoring-packets` to generate ignored worksheets with exact evidence snapshots.
3. To reproduce the starting points, run `seed-pilot-candidates --overwrite`. The command refuses to replace existing candidates unless `--overwrite` is explicit, because those files may contain human edits. Otherwise, author/revise complete candidate envelopes here with `status=candidate` and no reviews.
4. Run `audit-pilot-candidates` and `build-candidate-review-packets`.
5. Build the recruitment kit, register real reviewers, and run `audit-reviewers`. A `general` qualification never covers a sensitive category.
6. Run `assign-pilot-reviewers` and `export-assigned-review-kits`, then collect blinded decisions in `data/pilot/reviews.jsonl` using `schemas/pilot-review.schema.json`.
7. Record any disagreement in `data/pilot/adjudications.jsonl`. A revised candidate receives a new revision and fresh blinded reviews; adjudication cannot directly convert a disputed version into accepted data.
8. Run `validate-review-ledger`, then `finalize-reviewed-pilot`. Only that command writes accepted split files and updates their manifest hashes and counts.
9. Run `pilot-preflight`, `materialize-pilot`, and only then the smoke-test-only Lightning configuration.

Never invent reviewer identities or copy one review into another reviewer’s lane.
