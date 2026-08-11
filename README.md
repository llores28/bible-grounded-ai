# Bible-Grounded AI Initiative

An open research initiative to develop a Bible-grounded moral reasoning and life-guidance AI with biblical-language scholarship, transparent theological assumptions, evidence-backed interpretation, and strong safeguards for human dignity and safety.

This repository contains the technical plan, governance policies, deterministic Python safety and verification core, QLoRA/DPO configuration, evaluation scaffolding, and frontend source for the public project website.

## Project goals

- Evaluate Apertus 1.5B, 4B, and 8B candidates, then adapt the smallest model that passes the interpretation and safety capacity gates.
- Build retrieval-backed manuscript and Scripture citation workflows.
- Build an auditable "Scripture interprets Scripture" evidence graph for quotations, allusions, typology, symbols, prophetic durations, and moral application.
- Evaluate historicist prophetic rules—including the 42-month/1,260-day correspondence and day-year application—with explicit arithmetic, assumptions, alternative readings, and anti-numerology safeguards.
- Define a transparent Biblical Moral Constitution for practical reasoning.
- Distinguish explicit text, multi-passage canonical synthesis, historical interpretation, organizational alignment, and speculation without favoring or penalizing a doctrine because a denomination teaches it.
- Validate linguistic accuracy, citation reliability, safety, and general-capability retention before making public claims.

The system is not intended to claim divine authority, moral consciousness, or replacement of Scripture, prayer, conscience, pastoral care, or qualified professional advice.

## Repository contents

- [`MORAL_CONSTITUTION.md`](MORAL_CONSTITUTION.md), [`HERMENEUTICS_POLICY.md`](HERMENEUTICS_POLICY.md), [`THEOLOGY_POLICY.md`](THEOLOGY_POLICY.md), and [`SAFETY_POLICY.md`](SAFETY_POLICY.md) define the normative and interpretive controls.
- [`configs/commandments.json`](configs/commandments.json) and [`PROPHETIC_RULE_REGISTRY.yaml`](PROPHETIC_RULE_REGISTRY.yaml) are the executable commandment and prophetic policies.
- [`src/biblical_moral_ai/`](src/biblical_moral_ai/) implements approved-source retrieval, canonical graph storage, exact citation checking, safe arithmetic, commandment and pastoral-safety checks, local inference, dataset validation, training preflight, and release gates.
- [`schemas/`](schemas/) defines the machine-readable answer, review, corpus, preference, and release contracts.
- [`configs/training/`](configs/training/) contains pinned Apertus 8B QLoRA SFT and DPO experiment configurations.
- [`configs/data/source_packages.json`](configs/data/source_packages.json) and [`configs/data/lexicon_packages.json`](configs/data/lexicon_packages.json) lock the approved Scripture, morphology, Hebrew/Aramaic dictionary, and Koine Greek dictionary inputs.
- [`evals/`](evals/) contains public adversarial cases, a sealed-set custody contract, and release-metric templates.
- [`docs/Apertus_Bible_Grounded_AI_Master_Plan.md`](docs/Apertus_Bible_Grounded_AI_Master_Plan.md) is the v1.3 technical roadmap; [`docs/TRAINING_RUNBOOK.md`](docs/TRAINING_RUNBOOK.md) is the fail-closed CUDA training procedure.
- [`docs/PILOT_REVIEW_WORKFLOW.md`](docs/PILOT_REVIEW_WORKFLOW.md) defines the 50/20/25 authoring queue, blinded human review ledger, revision/adjudication rules, and CPU validation receipt.
- [`app/`](app/) and [`public/`](public/) contain the public project website.

## Validate the implementation

Requirements: Python 3.11 or newer. The deterministic core has no runtime dependencies outside the standard library.

```powershell
$env:PYTHONPATH = "src"
python -m biblical_moral_ai validate
python -m ruff check src tests/python
python -m unittest discover -s tests/python -v
```

Build the approved, digest-verified evidence store without committing third-party corpora:

```powershell
python -m biblical_moral_ai build-evidence --fetch
python -m biblical_moral_ai audit-pilot-drafts
python -m biblical_moral_ai build-authoring-packets
python -m biblical_moral_ai audit-pilot-candidates
python -m biblical_moral_ai search-lexicon "חֶסֶד" --language Hebrew
python -m biblical_moral_ai search-lexicon "ἀγάπη" --language "Koine Greek"
```

The pilot and production training preflights intentionally remain separate. Both currently return exit code `2` because real independently reviewed datasets do not yet exist:

```powershell
python -m biblical_moral_ai pilot-preflight
python -m biblical_moral_ai preflight
python -m biblical_moral_ai train configs/training/apertus_8b_qlora_pilot.json
```

See [`docs/TRAINING_RUNBOOK.md`](docs/TRAINING_RUNBOOK.md) before installing optional CUDA dependencies or executing a training run.

## Website

Public project site: [bible-grounded-ai.llores28.chatgpt.site](https://bible-grounded-ai.llores28.chatgpt.site)

## Run the frontend locally

Requirements: Node.js 22.13 or newer on Linux.

```bash
npm ci
npm run dev
```

Useful checks:

```bash
npm run lint
npm test
```

## Current status

The approved Scripture and linguistic sources are pinned and the fail-closed import pipeline is implemented. A curated, evidence-resolved `draft_only` queue and deterministic AI-authored candidate set contain 50 SFT, 20 preference, and 25 evaluation scenarios and pass CPU candidate validation. Accepted counts remain `0/50`, `0/20`, and `0/25` until real independent reviewers approve them. Production remains `0/3,000` SFT and `0/1,000` preference pairs. No adapter has been trained and no scholar-facing release is authorized. See [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md).

## Contributions

Early contributions are most useful in dataset provenance and licensing, biblical-language review, theological evaluation, ML training efficiency, safety testing, and reproducible evaluation design.

Before contributing copyrighted datasets, translations, lexicons, manuscript transcriptions, or scholarly materials, verify that the license permits the intended use and preserve source-level provenance.

## License

No open-source license has been selected yet. Until a license is added, copyright remains with the repository owner. Third-party dependencies and referenced datasets retain their own licenses.
