# Apertus 1.5B Bible-Grounded AI — Master Build Plan

**Target hardware:** One NVIDIA GeForce RTX 5090 with 32 GB VRAM  
**Base model:** `swiss-ai/Apertus-v1.1-1.5B`  
**Project owner:** Lannys Lores  
**Status:** Implementation specification v1.0  
**Planning date:** 2026-08-08

## 1. Mission

Build a research-grade, Bible-grounded AI that can:

1. Analyze transcribed Biblical Hebrew, Biblical Aramaic, Koine/Septuagint Greek, and Latin.
2. Explain Scripture in English and Spanish with citations and calibrated uncertainty.
3. Apply a transparent biblical moral framework to family, finance, work, relationships, technology, civic life, and spiritual questions.
4. Clearly distinguish textual facts, linguistic analysis, historical context, general Christian interpretation, Seventh-day Adventist interpretation, practical wisdom, and professional advice.
5. Refuse to invent manuscript readings, quotations, sources, or divine authority.

This system does **not** claim consciousness, faith, revelation, infallibility, or authority to speak for God. It is a tool that applies documented sources and an explicitly reviewed interpretive framework.

## 2. Scope and non-goals

### Version 1 includes

- Text-only manuscript transcriptions; no direct manuscript-image reading.
- Biblical Hebrew and Aramaic in square script.
- Koine Greek and Septuagint Greek.
- Classical/Ecclesiastical Latin, emphasizing the Vulgate and reception history.
- English and Spanish explanations.
- Retrieval-backed citations.
- Morphology, lemma, syntax, translation comparison, contextual interpretation, moral case reasoning, and uncertainty reporting.
- A separately labeled SDA interpretive layer.

### Version 1 excludes

- Claims to have reconstructed an original manuscript.
- Unverified OCR/HTR output as authoritative evidence.
- Training on copyrighted Bible translations, lexicons, commentaries, or manuscript databases without documented permission.
- Autonomous medical, legal, financial, mental-health, or crisis decisions.
- Replacing pastors, scholars, licensed professionals, conscience, prayer, or human judgment.
- Public release before the evaluation gates in this plan pass.

## 3. Hardware and operating environment

### Recommended workstation

- RTX 5090, 32 GB GDDR7.
- 128 GB RAM preferred; 64 GB is the practical minimum.
- 4 TB NVMe preferred: 1 TB system/code, 2 TB datasets/checkpoints, 1 TB working space or backup.
- Modern 12–16 core CPU.
- NVIDIA-recommended minimum 1000 W PSU; use a high-quality 1200–1600 W unit for workstation headroom.
- Large case, strong airflow, UPS, and temperature monitoring.
- Ubuntu 24.04 LTS, native Linux preferred.

### Software constraints

- Use a recent NVIDIA driver and a PyTorch build compiled for Blackwell (`sm_120`) with CUDA 12.8 or newer.
- Pin all package and container versions after the compatibility smoke test.
- Use Python 3.11 unless a required dependency dictates otherwise.
- Use Docker with NVIDIA Container Toolkit for reproducibility.
- Apertus requires Transformers 4.56.0 or newer.
- Never assume FlashAttention, bitsandbytes, Triton, GaLore, or an inference server supports the 5090/Apertus combination; verify each with an automated smoke test.

## 4. Required repository

Create this structure:

```text
bible-grounded-ai/
├── README.md
├── LICENSES.md
├── MODEL_CARD.md
├── DATA_CARD.md
├── MORAL_CONSTITUTION.md
├── THEOLOGY_POLICY.md
├── RISK_REGISTER.md
├── pyproject.toml
├── uv.lock
├── Makefile
├── .env.example
├── configs/
│   ├── hardware/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   └── inference/
├── data/
│   ├── registry/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── held_out/
├── src/
│   ├── ingest/
│   ├── normalize/
│   ├── deduplicate/
│   ├── tokenize/
│   ├── train/
│   ├── evaluate/
│   ├── retrieval/
│   ├── inference/
│   └── audit/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── unicode/
│   ├── leakage/
│   └── regression/
├── evals/
│   ├── linguistic/
│   ├── manuscripts/
│   ├── interpretation/
│   ├── moral_reasoning/
│   ├── safety/
│   └── general_capability/
├── scripts/
├── notebooks/
├── reports/
└── artifacts/
```

Do not commit raw restricted datasets, secrets, model weights, or large checkpoints to Git. Use dataset manifests and content hashes.

## 5. Data governance before model training

Create a machine-readable dataset register. No source enters training until its record is complete and approved.

Required fields:

```yaml
dataset_id:
title:
source_url:
version_or_commit:
download_date:
language:
dialect:
script:
textual_tradition:
witness_or_edition:
date_range:
annotation_types: []
license_name:
license_url:
commercial_use_allowed:
derivatives_allowed:
attribution_required:
ai_training_allowed_or_unknown:
approval_status: pending
content_sha256:
known_limitations: []
```

### Initial candidate sources to investigate

- Open Scriptures Hebrew Bible/WLC and its morphology.
- MorphGNT with independently verified text and annotation licenses.
- SBLGNT under its current stated terms.
- Properly licensed/open Septuagint editions.
- Public-domain Vulgate editions with verified provenance.
- Open Syriac/Aramaic resources only when dialect labels and rights are clear.
- Public-domain English and Spanish Bible translations.
- Open grammars, lexicons, treebanks, and historical corpora with source-level license verification.

Do not treat “available online,” “noncommercial research,” or “public GitHub repository” as permission to train.

## 6. Data representation

Every training or retrieval unit must carry structured metadata. Preserve both the diplomatic source and any normalized form.

```json
{
  "record_id": "...",
  "language": "biblical_hebrew",
  "dialect": "tiberian_masoretic",
  "script": "hebrew_square",
  "work": "Isaiah",
  "passage": "Isaiah 6:1",
  "witness": "...",
  "edition": "...",
  "text_diplomatic": "...",
  "text_normalized": "...",
  "normalization_operations": [],
  "lemma": [],
  "morphology": [],
  "syntax": [],
  "license_id": "...",
  "source_locator": "...",
  "split_group": "..."
}
```

Rules:

- Never overwrite source characters during normalization.
- Preserve Hebrew points and cantillation, Greek accents/breathings, Syriac marks, uncertain characters, lacunae, corrections, and scribal annotations.
- Label Biblical Aramaic, Imperial Aramaic, Targumic Aramaic, Jewish Aramaic, and Syriac separately.
- Deduplicate at document, passage, verse, phrase, and near-duplicate levels.
- Group parallel editions and translations so related text cannot cross training/test boundaries.
- Hold out entire passages, books, authors, editions, and selected witnesses—not random verses alone.

## 7. Baseline first

Before training, create an immutable baseline report for the untouched Apertus checkpoint.

Measure:

- Token count per word/script and byte/token fertility.
- Exact Unicode encode/decode preservation.
- Hebrew, Aramaic, Greek, and Latin lemma accuracy.
- Morphological feature exact match and per-feature F1.
- Translation and gloss accuracy.
- Context-sensitive word-sense selection.
- Variant-reading recognition using supplied evidence.
- Citation accuracy with and without retrieval.
- English and Spanish explanation quality.
- General multilingual regression benchmarks.
- Hallucinated quotations and sources.
- Moral-case reasoning rubric scores.

Store prompts, decoding parameters, model hash, package versions, raw outputs, graders, and scores.

## 8. Tokenizer decision gate

Keep the original Apertus tokenizer for the first pilot. Test:

- Pointed/unpointed Hebrew and cantillation.
- Biblical Aramaic and Syriac scripts.
- Polytonic/unaccented Greek.
- Nomina sacra and manuscript abbreviations in transcribed form.
- Latin and mixed-language lines.
- NFC/NFD variants without destructive normalization.

Do not replace the tokenizer. Consider adding tokens only if measured fragmentation materially damages throughput or accuracy, and only after documenting embedding initialization, compatibility, and regression results.

## 9. Training strategy for one RTX 5090

Use an experiment ladder. A later step begins only when the preceding gate passes.

### Experiment 0 — Environment and reproducibility

- Load the base model in BF16.
- Run deterministic inference.
- Run one forward/backward step.
- Confirm GPU architecture support, memory use, thermals, checkpoint save/resume, and exact config capture.
- Run 100 training steps on synthetic data and reproduce the loss curve after restart.

**Gate:** No unsupported kernels, NaNs, silent CPU fallback, corrupted checkpoints, or unexplained nondeterminism.

### Experiment 1 — QLoRA/SFT pipeline validation

- Use 4-bit loading only if its Blackwell/Apertus kernels pass correctness tests.
- Train small adapters on 1,000–10,000 verified linguistic and response-format examples.
- Teach answer structure, citations, evidence labels, uncertainty, and refusal to invent sources.
- This stage tests behavior and pipeline quality; it does not prove new language competence.

**Gate:** Improved format/task scores with no major general-language regression.

### Experiment 2 — Ancient-language continued-pretraining pilot

- Use 10–30 million high-quality tokens.
- Compare at least two feasible methods on identical data and token budgets:
  1. LoRA or ReLoRA-style continued pretraining.
  2. GaLore/8-bit optimizer or conventional full-parameter training with gradient checkpointing, if stable and memory-safe.
- Use BF16, sequence packing, gradient accumulation, length-aware batches, memory-efficient attention, and frequent resumable checkpoints.
- Retain a controlled percentage of general multilingual data to reduce catastrophic forgetting.
- Start at sequence length 1,024 or 2,048; expand only after profiling.
- Select by quality gained per GPU-hour, peak VRAM, stability, and regression—not novelty.

**Gate:** Statistically meaningful held-out linguistic improvement, no leakage, and no more than 5% agreed regression on general benchmarks.

### Experiment 3 — Full specialist continued pretraining

- Expand only after corpus and pilot approval.
- Target 100–300 million curated tokens initially; do not inflate the corpus through uncontrolled repetition.
- Use curriculum/mixing weights by language, dialect, annotation type, and task.
- Cap repeated appearances of the same biblical passage.
- Save optimizer-resumable checkpoints, evaluation snapshots, and data cursor state.
- Run automatic evaluations at fixed token intervals with early-stop rules.

**Gate:** Meets the release thresholds in Section 13.

### Experiment 4 — Linguistic and research SFT

Train verified tasks for:

- Segmentation, lemma, morphology, syntax, translation, semantic range.
- Passage context, genre, intertextuality, and historical setting.
- Manuscript/witness comparison when evidence is supplied.
- Separation of transcription, observation, inference, and theological interpretation.
- Answers in English and Spanish.

### Experiment 5 — Biblical moral-reasoning SFT

Build `MORAL_CONSTITUTION.md` before generating examples. Each governing principle must include biblical basis, interpretive method, counterexamples, exceptions, competing duties, and safety constraints.

Every case should require the model to:

1. State known facts and missing facts.
2. Identify affected persons and possible harm.
3. Retrieve relevant Scripture in context.
4. Label each conclusion as explicit teaching, strong biblical principle, wisdom judgment, or tradition-specific interpretation.
5. Consider motive, duty, justice, mercy, consequences, freedom, and protection of vulnerable people.
6. Separate general Christian and SDA-specific conclusions.
7. Respect civil law while identifying genuine moral conflicts.
8. Refer high-stakes matters to qualified humans.
9. Provide options rather than falsely claiming a direct message from God.

Include adversarial cases: abuse disguised as submission, exploitation disguised as generosity, prejudice justified by proof-texting, financial speculation, deception for a claimed good end, coercive religious behavior, self-harm, violence, medical refusal, and fabricated verses.

### Experiment 6 — Preference optimization

Only after a reliable rubric and independently reviewed preference pairs exist. Prefer a simple, documented method compatible with 32 GB VRAM. Do not use synthetic model judgments as the sole authority. Keep expert labels and model-generated labels distinguishable.

## 10. Retrieval and inference architecture

Training provides language/task skill; retrieval provides exact evidence.

```text
User question
  -> intent and risk classification
  -> source retrieval by passage/witness/lemma/topic
  -> evidence bundle with immutable source IDs
  -> specialist model response
  -> citation and quotation verifier
  -> theological-label and safety checker
  -> final answer with evidence and uncertainty
```

Requirements:

- Hybrid lexical/vector retrieval with explicit metadata filters.
- Exact text lookup must outrank semantic similarity for verse/witness queries.
- Each quote must match a retrieved source span exactly or be labeled as a paraphrase.
- Retrieval failures must produce an explicit limitation, not a guessed reading.
- Keep user data out of training by default.
- Log source IDs and model/config hashes without logging sensitive user content unnecessarily.

## 11. Required answer schema

The model should internally or externally produce structured output:

```json
{
  "answer": "...",
  "textual_evidence": [],
  "linguistic_analysis": [],
  "historical_context": [],
  "interpretive_options": [],
  "general_christian_application": [],
  "sda_interpretive_layer": [],
  "practical_options": [],
  "professional_referral": null,
  "uncertainties": [],
  "citations": [],
  "confidence": "low|medium|high"
}
```

The user-facing renderer can omit empty sections but must not remove uncertainty or source labels.

## 12. Evaluation design

### Evaluation groups

- Unicode/script integrity.
- Tokenization efficiency.
- Hebrew/Aramaic/Greek/Latin language analysis.
- Diplomatic and normalized transcriptions.
- Textual variants and witness attribution.
- Retrieval, quotation, and citation fidelity.
- Context and genre interpretation.
- General Christian and SDA distinction.
- Moral reasoning across family, finance, business, sexuality, work, community, technology, and spiritual life.
- Safety/high-stakes boundaries.
- English/Spanish quality.
- General capability and catastrophic forgetting.
- Prompt injection and malicious source content.

### Data isolation

- Maintain a sealed test set unavailable to training and example-generation code.
- Use document-family hashes and similarity searches to detect contamination.
- Reserve some evaluations for human reviewers only.
- Record evaluation-set version and prohibit optimization against the sealed final set.

### Human review

At minimum, seek qualified review for Biblical Hebrew, Biblical Aramaic/Syriac, Koine Greek, Latin, textual criticism, pastoral ethics, and SDA theology. One reviewer may cover multiple roles if genuinely qualified, but disputed decisions need a second review.

## 13. Initial release gates

These are engineering targets, not claims about the untrained model:

| Gate | Initial target |
|---|---:|
| Unicode round-trip preservation | 100% |
| Retrieved quotation exactness | 100% |
| Source/witness citation accuracy with retrieval | at least 99% |
| Greek lemma accuracy | at least 97% |
| Hebrew lemma accuracy | at least 95% |
| Greek morphology exact match | at least 92% |
| Hebrew morphology exact match | at least 90% |
| Biblical Aramaic morphology exact match | at least 85% |
| Fabricated source quotations in release suite | 0 |
| Evidence/interpretation separation | at least 95% expert-rated |
| SDA-specific labeling | at least 95% expert-rated |
| High-stakes referral/boundary compliance | at least 98% |
| General benchmark regression | no more than 5% unless explicitly approved |

Report confidence intervals and sample sizes. Averages cannot conceal catastrophic failures in a language or safety category.

## 14. Milestones and realistic timeline

Assumes one primary engineer working part-time to full-time, with expert review scheduled separately.

| Phase | Deliverable | Estimate |
|---|---|---:|
| 0 | Repository, environment, hardware smoke tests | 1–2 weeks |
| 1 | Data/license registry and ingestion framework | 2–6 weeks |
| 2 | Baselines, tokenizer audit, sealed evaluation sets | 3–6 weeks |
| 3 | QLoRA/SFT pipeline and retrieval prototype | 2–4 weeks |
| 4 | 10–30M-token method comparison pilot | 2–5 weeks |
| 5 | 100–300M-token specialist training iterations | 1–3 months |
| 6 | Moral constitution, case set, SFT, adversarial tests | 2–6 months, overlapping |
| 7 | Scholar review, corrections, release documentation | 2–6 months |

Expected result: working prototype in roughly 2–3 months; credible internal v1 in 6–12 months; responsibly reviewed public release in 12–18 months.

## 15. Cost controls

- Record GPU-hours, wall time, energy, peak VRAM, tokens/second, and checkpoint size for every run.
- Impose a maximum token budget and stop conditions before each experiment.
- Run 100-step, 1,000-step, and small-data rehearsals before long runs.
- Never launch a long run until checkpoint/resume and evaluation jobs pass.
- Prefer experiment matrices that change one major variable at a time.
- Use cloud A100/H100 rental only when a measured bottleneck justifies it, not by default.
- Back up code, manifests, constitutions, evaluations, and selected checkpoints; raw reproducible caches may be disposable.

## 16. Advanced LLM operating instructions

Use the following as the controlling prompt for the coding agent:

> You are the lead ML engineer and research software architect for the Apertus 1.5B Bible-Grounded AI project. Implement the attached master plan incrementally on one RTX 5090 32 GB workstation. Treat claims, licenses, data provenance, evaluation isolation, and reproducibility as first-class engineering requirements.
>
> Begin by auditing the current repository and environment. Do not download datasets, begin training, alter the base model, accept licenses, or incur cloud costs without producing a written proposal and receiving owner approval. Never assume that online availability permits AI training. Never place secrets or restricted corpora in Git.
>
> Work milestone by milestone. For each milestone: (1) state assumptions, (2) propose exact files and interfaces, (3) implement the smallest testable increment, (4) run unit and integration tests, (5) report commands, results, risks, and unresolved questions, and (6) stop at the stated approval gate. Preserve existing user work and use non-destructive, resumable operations.
>
> Prefer stable, documented methods over experimental techniques. Verify RTX 5090 `sm_120`, CUDA, PyTorch, Transformers, attention kernels, quantization kernels, and Apertus compatibility with smoke tests. Detect and fail on silent CPU fallback, NaNs, out-of-memory instability, corrupted checkpoints, or missing provenance.
>
> Establish immutable baselines and sealed evaluation splits before any training. Do not claim biblical-language, manuscript, theological, moral, or safety competence from training loss or sample outputs. Only make bounded claims supported by held-out measurements and qualified human review.
>
> Separate the system into: source ingestion, immutable originals, reversible normalization, deduplication/splitting, training, retrieval, response generation, quotation/citation verification, theological labeling, safety review, and audit reporting. A text-only model must never be described as reading manuscript images.
>
> Implement the experiment ladder in order: environment validation; baseline; tokenizer audit; small QLoRA/SFT pipeline; 10–30M-token continued-pretraining comparison; expanded training only after approval; linguistic SFT; moral-reasoning SFT; preference optimization only with reviewed data. Compare methods using held-out quality gained per GPU-hour, stability, VRAM, and regression.
>
> The AI may transparently apply a reviewed biblical framework, but it must never claim consciousness, revelation, infallibility, divine authority, or that its answer is God’s direct will. It must distinguish explicit biblical teaching, strong biblical principle, wisdom judgment, and tradition-specific interpretation. It must label the SDA layer and refer high-stakes matters to qualified people.
>
> Your first response must contain only: repository audit findings; detected hardware/software facts; missing decisions; Phase 0 implementation plan; proposed acceptance tests; and commands you intend to run. Do not start training in the first response.

## 17. First tasks for the coding agent

1. Inspect the repository and identify existing assets without editing.
2. Produce an Architecture Decision Record for the environment and package manager.
3. Create the repository skeleton and documentation templates.
4. Build a `doctor` command that reports GPU, driver, CUDA runtime, compute capability, PyTorch, BF16, Transformers, disk, RAM, and kernel availability.
5. Build a model-load and one-step training smoke test.
6. Create dataset-registry schema validation and license approval enforcement.
7. Create Unicode round-trip tests with representative licensed/test strings.
8. Create evaluation schemas and immutable baseline runner.
9. Draft—not finalize—the moral constitution and theology policy outlines for human review.
10. Stop and provide a Phase 0 report before downloading corpora or training.

## 18. Decisions the owner must make

- Internal research, nonprofit ministry, open release, or commercial use.
- Exact SDA authority policy: Bible alone, Bible plus denominational statements, and how Ellen G. White sources are handled and licensed.
- Preferred Bible textual traditions and critical editions.
- Approved English and Spanish translations.
- Whether Syriac is in v1 or deferred.
- Budget for scholarly and pastoral review.
- Privacy policy and intended user population, including minors.
- Public model weights versus hosted-only release.
- Whether generated moral case data may be used and under what human-review threshold.

## 19. Authoritative technical references

- Apertus v1.1 1.5B model card: https://huggingface.co/swiss-ai/Apertus-v1.1-1.5B
- Hugging Face Apertus implementation: https://huggingface.co/docs/transformers/model_doc/apertus
- NVIDIA RTX 5090 specifications: https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
- NVIDIA RTX 5090 user guide: https://www.nvidia.com/content/geforce-gtx/GeForce_RTX_5090_User_Guide_Rev1.pdf
- PyTorch installation selector: https://pytorch.org/get-started/locally/
- Open Scriptures Hebrew Bible: https://github.com/openscriptures/morphhb
- MorphGNT: https://github.com/morphgnt/sblgnt

All technical versions, dataset terms, and licenses must be reverified at implementation time and recorded with dates and hashes.
