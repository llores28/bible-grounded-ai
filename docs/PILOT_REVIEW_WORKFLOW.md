# Pilot Authoring and Independent Review Workflow

This workflow produces the 50 SFT, 20 preference, and 25 evaluation records required for the QLoRA smoke-test pilot. It does not weaken the separate 3,000/1,000 production gate.

## Trust boundary

The committed draft queue is an authoring plan, not training data. Generated worksheets, model-written answers, author self-reviews, copied reviews, invented reviewer identities, and unadjudicated disagreements count as zero accepted records.

Only real reviewers registered in `configs/reviewers.json` may approve candidates. Each reviewer records affiliation disclosures, qualified categories, and a dated independence attestation. A record author cannot review the same record. First-pass decisions are made independently without seeing another reviewer's decision.

## 1. Reproduce evidence and audit the queue

```powershell
$env:PYTHONPATH = "src"
python -m biblical_moral_ai build-evidence --fetch
python -m biblical_moral_ai audit-pilot-drafts
python -m biblical_moral_ai build-authoring-packets
```

The queue contains exactly 50 SFT, 20 preference, and 25 evaluation prompts. Every item is bound to an approved source and a passage that exists in the digest-matched evidence store. The ignored authoring packet file contains the exact quotation, source revision, canonical digest, license decision, and attribution.

## 2. Author complete candidates

Follow `data/pilot/candidates/README.md` and `schemas/pilot-candidate-envelope.schema.json`. Candidate records use `status=candidate`, contain no reviews, preserve the curated prompt/category/reference, and include exact source-license decision IDs.

The repository includes deterministic AI-authored starting candidates. Regenerate them only deliberately; `--overwrite` can destroy human edits:

```powershell
python -m biblical_moral_ai seed-pilot-candidates --overwrite
python -m biblical_moral_ai audit-pilot-candidates
python -m biblical_moral_ai build-candidate-review-packets
```

The candidate audit runs the deterministic commandment, safety, source approval, provenance, and exact-citation pipeline. The rejected answer in a preference pair may intentionally fail policy, but it must remain structurally valid. The chosen answer must produce the curated expected decision, including escalation for an imminent-danger scenario. Passing this audit is not human approval.

## 3. Assign and collect blinded reviews

Populate `configs/reviewers.json` with real reviewers. Do not publish information beyond what each reviewer consents to disclose.

Create the recruitment kit and inspect the exact capacity gaps before registering anyone:

```powershell
python -m biblical_moral_ai build-reviewer-recruitment-kit
python -m biblical_moral_ai audit-reviewers
```

The recruitment kit includes a copy-ready reviewer call, a qualification rubric, a blinded calibration plan, and `review-inventory.csv`. The inventory identifies all 95 candidate locations and the required qualification lane and independent decision count for each item. It is a handoff aid, not evidence of approval.

`general` qualifies a reviewer only for non-sensitive cases. Prophecy, abuse, violence, force, and disputed doctrine each require an explicit matching qualification. Two qualified reviewers are required for those lanes; a third qualified reviewer is recommended so a disagreement can be adjudicated by someone who did not submit a first-pass decision.

```powershell
python -m biblical_moral_ai assign-pilot-reviewers
python -m biblical_moral_ai export-assigned-review-kits
```

The export command creates a separate deterministic ZIP for each assigned reviewer under the ignored `data/reviewer_kits/` directory. Each bundle contains only that reviewer's packet set and blank decision templates; it contains no other reviewer's decisions. Preference pairs and all prophecy, abuse, violence, force, and disputed-doctrine cases require two independent reviewers. Other SFT/evaluation candidates require at least one. Completed review decisions go in `data/pilot/reviews.jsonl` and follow `schemas/pilot-review.schema.json`. Every decision binds to the exact `packet_sha256`; changing a candidate invalidates its reviews.

## 4. Adjudicate without laundering disagreement

If reviewers do not unanimously approve, a distinct active reviewer records an adjudication under `schemas/pilot-adjudication.schema.json`. An adjudication may require revision or reject the candidate. It cannot convert a disputed version directly into accepted data. A revision increments `candidate_revision`, receives a new packet digest, and undergoes fresh blinded review.

```powershell
python -m biblical_moral_ai validate-review-ledger
```

## 5. Finalize, hash, and validate on CPU

After unanimous approval of every current candidate:

```powershell
python -m biblical_moral_ai finalize-reviewed-pilot
python -m biblical_moral_ai pilot-preflight
python -m biblical_moral_ai write-pilot-audit-receipt
python -m biblical_moral_ai materialize-pilot
```

Finalization converts review decisions into the accepted record contracts, reruns the CPU validators and exact citation checks, writes all three accepted JSONL splits, and updates counts and SHA-256 values in `data/registry/pilot_manifest.json`. The manifest is written last, so an interrupted run fails closed.

The audit receipt binds the git commit, source registry, reviewer registry, draft queue, evidence database, generated packet files, validation results, and preflight result.

## 6. Lightning L4 smoke test

Do not open or spend Lightning resources until `pilot-preflight` returns ready. Then use the configured project Studio and run only:

```powershell
python -m biblical_moral_ai train configs/training/apertus_8b_qlora_pilot.json --execute --smoke-test
```

The pilot configuration refuses non-smoke execution. A successful two-step run proves pipeline execution only; it does not establish model quality, production readiness, or authorization to publish weights.
