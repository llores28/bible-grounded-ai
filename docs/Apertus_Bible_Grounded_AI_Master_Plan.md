# Apertus Bible-Grounded AI — Master Build Plan

**Target hardware:** One NVIDIA GeForce RTX 5090 with 32 GB VRAM  
**Initial training target:** `swiss-ai/Apertus-8B-Instruct-2509` at pinned revision; Apertus v1.5 8B remains a multimodal compatibility candidate
**Project owner:** Lannys Lores  
**Status:** Core implementation v1.3; data collection and training blocked by acceptance gates
**Planning date:** 2026-08-11

## 0. Implementation checkpoint

The v1.3 repository implements the moral constitution, hermeneutics and theology policies, commandment and prophetic registries, JSON contracts, approved-source SQLite retrieval, canonical graph constraints, exact citation verification, deterministic prophetic arithmetic, pastoral-safety checks, local inference gating, reviewed-data validators, QLoRA/DPO launchers, CUDA preflight, public adversarial cases, sealed-set custody rules, and non-waivable release metrics.

This is not a trained-model milestone. All biblical source records remain unapproved, accepted SFT and preference counts remain zero, and the sealed set has not been created. The code intentionally blocks training and release until those facts change through documented review. `IMPLEMENTATION_STATUS.md`, `DATA_CARD.md`, and `MODEL_CARD.md` are the controlling status disclosures.

## 1. Mission

Build a research-grade, Bible-grounded AI that can:

1. Analyze transcribed Biblical Hebrew, Biblical Aramaic, Koine/Septuagint Greek, and Latin.
2. Explain Scripture in English and Spanish with citations and calibrated uncertainty.
3. Trace how biblical passages interpret, reuse, quote, echo, or clarify other passages through an auditable canonical cross-reference graph.
4. Apply a transparent biblical moral framework to family, finance, work, relationships, technology, civic life, and spiritual questions.
5. Analyze prophetic structures, symbols, numbers, and timelines with explicit arithmetic, assumptions, source passages, and interpretive-school labels.
6. Clearly distinguish textual facts, linguistic analysis, canonical synthesis, historical context, organizational belief claims, practical wisdom, and professional advice.
7. Refuse to invent manuscript readings, quotations, sources, hidden codes, or divine authority.

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
- A reviewed "Scripture interprets Scripture" method using explicit quotation, allusion, parallel, type/antitype, symbol-definition, and thematic edges.
- Prophetic-time and symbolic-number analysis that exposes every premise and labels historicist, preterist, futurist, and idealist readings when relevant.
- Neutral organizational-alignment metadata that can report whether a church's official documents affirm a conclusion, without using the organization as evidence that the conclusion is biblically true.

### Version 1 excludes

- Claims to have reconstructed an original manuscript.
- Unverified OCR/HTR output as authoritative evidence.
- Training on copyrighted Bible translations, lexicons, commentaries, or manuscript databases without documented permission.
- Autonomous medical, legal, financial, mental-health, or crisis decisions.
- Replacing pastors, scholars, licensed professionals, conscience, prayer, or human judgment.
- Unbounded numerology, Bible-code searches, gematria-based claims without an approved policy, or patterns selected after seeing a desired result.
- Treating a historicist rule, denominational conclusion, proposed date anchor, or inferred symbol as if the biblical text explicitly states it.
- Treating an SDA or other religious organization as theological authority, using its publications as substitutes for biblical evidence, or labeling a well-supported canonical synthesis as speculative merely because that organization also teaches it.
- Predicting new dates for Christ's return or presenting speculative prophetic timelines as certain.
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
├── HERMENEUTICS_POLICY.md
├── PROPHETIC_RULE_REGISTRY.yaml
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
│   ├── hermeneutics/
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
│   ├── prophetic_reasoning/
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

Interpretive examples and graph edges require a second, evidence-bearing schema. A conclusion without this ledger is not eligible for SFT or preference data.

```json
{
  "claim_id": "prophetic_time_1260_equivalence",
  "claim_text": "The linked apocalyptic passages present 42 months, 1,260 days, and time-times-half-a-time as corresponding periods.",
  "claim_class": "textual_observation|cross_reference|arithmetic_inference|historicist_rule|moral_application",
  "supporting_passages": ["Revelation 11:2-3", "Revelation 12:6,14", "Revelation 13:5", "Daniel 7:25"],
  "edge_types": ["parallel_duration", "later_scripture_reuses_earlier_symbol"],
  "calculation": "42 * 30 = 1260; 3.5 * 360 = 1260",
  "assumptions": ["the passages describe corresponding symbolic periods", "month is treated schematically as 30 days"],
  "interpretive_school": "textual_observation|canonical_synthesis|historicist|preterist|futurist|idealist|other_named_method",
  "organizational_alignment": [
    {
      "organization": "Seventh-day Adventist Church",
      "official_statement_id": "fundamental_belief_20",
      "relationship": "officially_affirms|partially_affirms|does_not_address|disagrees",
      "source_locator": "https://adventist.org/en/beliefs"
    }
  ],
  "counter_readings": [],
  "historical_anchor_sources": [],
  "review_status": "pending|single_review|dual_review|approved|rejected",
  "reviewer_roles": [],
  "confidence": "low|medium|high"
}
```

Rules:

- Store observation, inference, and application as separate records; never collapse them into one target answer.
- Store organizational alignment separately from the evidence and interpretation fields. Organizational agreement cannot increase biblical-evidence confidence.
- Every cross-reference edge must name its type and point to exact source spans.
- Every numeric result must include a machine-checkable expression, units, rounding policy, and assumptions.
- A proposed historical fulfillment requires date-source provenance and at least two independent historical sources before it can be used as a positive training target.
- Preserve rejected and disputed claims for contrastive evaluation; do not silently delete them from the audit trail.

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
- Canonical cross-reference edge precision and recall by edge type.
- Prophetic arithmetic accuracy, unit consistency, and assumption disclosure.
- Ability to distinguish textual equivalence from a historicist day-year application.
- False-positive rate for unsupported symbols, hidden meanings, and numeric patterns.
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

## 8A. Model-capacity decision gate

Do not commit the project to 1.5B merely because it is inexpensive. The initial implementation pins the stable text checkpoint `swiss-ai/Apertus-8B-Instruct-2509` for specialist QLoRA. Evaluate Apertus v1.1 1.5B/4B as routing or retrieval baselines and evaluate Apertus v1.5 8B as a future multimodal candidate only after its custom Transformers integration, quantization, PEFT, and target-CUDA compatibility pass the same smoke tests.

Use the same frozen evaluation set, retrieval bundle, prompt format, and decoding settings. Measure:

- Multi-hop cross-reference reasoning and resistance to unsupported connections.
- Exact execution of prophetic arithmetic and unit conversions.
- Interpretation-school separation and counter-reading quality.
- Moral-case reasoning, citation fidelity, English/Spanish quality, latency, and peak VRAM.
- Quality gained per GPU-hour for QLoRA SFT and, where feasible, continued-pretraining pilots.

The 1.5B model may remain a low-cost edge or classifier model, but it must not become the primary interpreter unless it meets the same critical release gates. Prefer the smallest model that passes; current evidence makes 8B the more credible primary candidate for multi-step interpretation, subject to measurement rather than assumption.

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
- Train the first countable adapter only after at least 3,000 expert-reviewed SFT examples pass provenance, licensing, citation, commandment, safety, deduplication, and split-isolation checks.
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
- Typed quotation, allusion, parallel, type/antitype, symbol-definition, and thematic cross-reference edges.
- Manuscript/witness comparison when evidence is supplied.
- Separation of transcription, observation, inference, and theological interpretation.
- Answers in English and Spanish.

### Experiment 5 — Biblical moral-reasoning SFT

Build `MORAL_CONSTITUTION.md` before generating examples. Each governing principle must include biblical basis, interpretive method, counterexamples, exceptions, competing duties, and safety constraints.

Train the method, not a bank of predetermined conclusions. Each example must include an evidence ledger, the strongest reasonable counter-reading, and a reviewer-approved explanation of why the preferred application follows. Generated cases may propose candidates, but no generated theological conclusion becomes a positive target without human approval. Do not use denominational affiliation as a positive or negative label.

Use this doctrine-evidence rubric:

- `explicit_text`: directly stated in the focal passage.
- `canonical_synthesis`: supported by multiple contextually compatible passages whose combined claim is stronger than any one proof text.
- `contextual_inference`: plausible from genre, language, and context but not directly stated.
- `named_historical_interpretation`: a traceable method such as historicism, not a synonym for speculation.
- `organizational_alignment`: a factual statement that an organization officially affirms the doctrine; this carries no independent evidentiary weight.
- `speculative_hypothesis`: lacks sufficient textual support or depends on unapproved assumptions.

Create reviewed canonical-synthesis case sets for at least:

1. **The seventh-day Sabbath.** Begin with Genesis 2:1-3; Exodus 20:8-11; Deuteronomy 5:12-15; Isaiah 56:5-7; Ezekiel 20:12,20; Mark 2:27-28; Luke 4:16; and Hebrews 4:1-11. Evaluate each component claim separately rather than treating the doctrine name as its own proof.
2. **Death and resurrection.** Begin with Psalm 146:3-4; Ecclesiastes 9:5-6,10; Daniel 12:2; John 5:28-29; John 11:11-14; 1 Corinthians 15:51-54; and 1 Thessalonians 4:13-17. Distinguish what each passage explicitly says from the resulting synthesis about consciousness, resurrection, and immortality.
3. **The three angels' messages.** Begin with Revelation 14:6-12 in its literary context, then evaluate proposed links to creation/worship, judgment, Babylon, commandments, faith, Daniel, and Revelation 12-18 through reviewed graph edges. Do not replace Revelation's text with a denominational summary.

The SDA 28 Fundamental Beliefs may be used to confirm that the SDA Church officially affirms these and other doctrines. They must not be used as the textual ground truth, as a shortcut around passage analysis, or as evidence that competing readings are false.

Every case should require the model to:

1. State known facts and missing facts.
2. Identify affected persons and possible harm.
3. Retrieve relevant Scripture in context.
4. Label each conclusion as explicit text, canonical synthesis, contextual inference, named historical interpretation, organizational alignment, wisdom judgment, or speculative hypothesis.
5. Consider motive, duty, justice, mercy, consequences, freedom, and protection of vulnerable people.
6. Separate the biblical evidence judgment from any statement about which organizations officially affirm it.
7. Respect civil law while identifying genuine moral conflicts.
8. Refer high-stakes matters to qualified humans.
9. Provide options rather than falsely claiming a direct message from God.
10. Identify when a moral conclusion depends on a disputed symbolic, typological, prophetic, or denominational premise.

Include adversarial cases: abuse disguised as submission, exploitation disguised as generosity, prejudice justified by proof-texting, financial speculation, deception for a claimed good end, coercive religious behavior, self-harm, violence, medical refusal, and fabricated verses.

### Experiment 6 — Preference optimization

Only after a reliable rubric and independently reviewed preference pairs exist. Prefer DPO or another simple, documented offline preference method compatible with 32 GB VRAM. Do not use synthetic model judgments as the sole authority. Keep expert labels and model-generated labels distinguishable. Preference pairs must reward textual grounding, faithful arithmetic, fair representation of alternatives, calibrated confidence, and refusal of unsupported hidden meanings—not agreement with an unlabeled doctrinal conclusion.

## 9A. Bible-interprets-Bible and prophetic-reasoning framework

`HERMENEUTICS_POLICY.md` is the controlling interpretation specification. Scripture is the primary evidence layer; lexicons, grammars, manuscript evidence, historical sources, and commentary are supporting layers whose roles must be labeled. The system may use later biblical passages to illuminate earlier ones, but it must preserve each passage's language, genre, local context, and historical setting.

### Evidence hierarchy

Use the following order and report when a conclusion moves down the hierarchy:

1. Exact wording and grammar of the focal passage.
2. Immediate literary context and book-level argument.
3. Explicit biblical quotation or author-identified fulfillment.
4. Strong lexical, thematic, typological, or structural parallel with traceable features.
5. Broader canonical synthesis.
6. Historical context and proposed fulfillment from independently sourced records.
7. Named historical interpretive method, with its premises stated and tested against the preceding evidence layers.
8. Speculative hypothesis, which cannot be presented as doctrine or used as a preferred SFT target without exceptional review.

Organizational belief documents are not part of this authority hierarchy. They may be retrieved only to answer the factual question of what an organization officially teaches or to compare that statement with an independently constructed biblical evidence ledger.

### Canonical evidence graph

Build a versioned graph whose nodes are passages, lexical senses, people, places, events, symbols, covenants, moral principles, and historical claims. Allow only reviewed edge types:

- `explicit_quote`
- `explicit_fulfillment`
- `verbal_allusion`
- `narrative_parallel`
- `type_antitype`
- `symbol_defined_by_text`
- `shared_duration`
- `shared_lexeme_or_phrase`
- `thematic_parallel`
- `contrast`
- `proposed_historical_fulfillment`

Every edge stores exact spans, direction, reviewer status, alternative explanations, and confidence. Retrieval must return the underlying passages with the edge; the graph itself is never sufficient evidence.

### Initial prophetic rule registry

Seed `PROPHETIC_RULE_REGISTRY.yaml` with candidate rules, not unquestionable axioms:

1. **Apocalyptic duration equivalence.** Revelation 11:2-3 presents 42 months and 1,260 days in adjacent descriptions; Revelation 12:6,14 presents 1,260 days and "time, times, and half a time" for the woman's wilderness period; Revelation 13:5 repeats 42 months. The arithmetic `42 * 30 = 1,260` and `3.5 * 360 = 1,260` is exact under a schematic 30-day month. Label this as a strong textual/arithmetic correspondence, while identifying the assumption that the linked expressions refer to corresponding periods.
2. **Day-for-year sign acts.** Numbers 14:34 and Ezekiel 4:6 explicitly assign a year for each symbolic day in their own contexts. Treat these as direct precedents. Applying the rule universally to Daniel or Revelation is a historicist inference and must be labeled rather than described as an explicit universal command.
3. **"Time" as a year.** Daniel 4's "seven times" is commonly read as seven years, and Daniel 7:25, 12:7, and Revelation 12:14 support a three-and-a-half-period correspondence. Store lexical form, context, and translation evidence. Do not assume every occurrence of Aramaic `iddan` or Greek `kairos` equals one year.
4. **Historicist day-year application.** The conversion of 1,260 prophetic days into 1,260 historical years is an important historicist rule. The model may apply it within that labeled framework, show the arithmetic, and test proposed anchors; it may not call the conversion an undisputed textual fact. Whether the SDA Church officially affirms this method is separate organizational metadata, not evidence for or against the rule.
5. **360-day prophetic year.** Treat 360 as a schematic apocalyptic year derived from the 42-month/1,260-day equivalence, not as proof that every biblical or ancient civil year always contained 360 days.

For each rule store: scope, source passages, lexical notes, calculation, assumptions, supporting arguments, objections, accepted schools, rejected uses, reviewer decisions, and version history.

### Prophetic timeline workflow

For every timeline question, require the model or orchestration layer to:

1. Identify the focal passage, genre, speaker, audience, and textual witnesses.
2. Retrieve direct parallels before broad thematic similarities.
3. List symbol definitions actually supplied by Scripture.
4. Select only approved rules whose scope matches the passage.
5. Execute arithmetic in a deterministic calculator, not free-form language generation.
6. Separate textual duration, symbolic conversion, start anchor, end anchor, and historical identification.
7. Cite primary historical evidence for proposed dates and report calendar/chronology uncertainty.
8. Generate historicist, preterist, futurist, and idealist summaries when the question is disputed.
9. State which conclusion follows under which assumptions and refuse false certainty.

### Hidden-lesson and number safeguards

- Require a pre-registered rule or an independently reviewable textual feature; coincidence alone is not evidence.
- Test whether a proposed pattern survives translation choice, verse-number removal, spelling variation, and comparison with suitable control texts.
- Penalize cherry-picked start/end points, unit switching, repeated arithmetic transformations, and patterns discovered only after selecting a desired result.
- Distinguish author-signaled symbolism from reader-generated association.
- Never use model confidence as evidence that a hidden meaning is real.
- Moral applications must be supportable without coercion, dehumanization, date-setting, or claims of private revelation.

## 10. Retrieval and inference architecture

Training provides language/task skill; retrieval provides exact evidence.

```text
User question
  -> intent and risk classification
  -> source retrieval by passage/witness/lemma/topic and reviewed graph edges
  -> evidence bundle with immutable source IDs, rule IDs, and alternative readings
  -> deterministic arithmetic/chronology tools when required
  -> specialist model response
  -> claim-level citation, quotation, and calculation verifier
  -> theological-label and safety checker
  -> final answer with evidence and uncertainty
```

Requirements:

- Hybrid lexical/vector retrieval with explicit metadata filters.
- Exact text lookup must outrank semantic similarity for verse/witness queries.
- Reviewed explicit-quotation and symbol-definition edges must outrank model-generated thematic similarities.
- Each quote must match a retrieved source span exactly or be labeled as a paraphrase.
- Each interpretive claim must resolve to source spans plus an approved rule ID or be labeled as an unapproved hypothesis.
- Never let the model calculate prophetic dates internally when a deterministic calculator and calendar-aware chronology module can do so.
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
  "canonical_cross_references": [],
  "hermeneutic_rules_applied": [],
  "calculations": [],
  "assumptions": [],
  "interpretive_options": [],
  "preferred_reading_under_stated_framework": null,
  "general_christian_application": [],
  "organizational_alignment": [],
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
- Canonical cross-reference classification and graph-edge fidelity.
- Prophetic duration arithmetic, units, calendars, and chronology.
- Day-year, time-as-year, type/antitype, and symbol-rule scope control.
- Historicist/preterist/futurist/idealist comparison without viewpoint leakage.
- Unsupported hidden-meaning and numerology rejection.
- Biblical support classification versus organizational-belief attribution.
- Sabbath, death/resurrection, and three-angels-message canonical synthesis without denominational shortcutting.
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

At minimum, seek qualified review for Biblical Hebrew, Biblical Aramaic/Syriac, Koine Greek, Latin, textual criticism, pastoral ethics, apocalyptic literature, history/chronology, and historicist interpretation. Select reviewers for demonstrated subject competence rather than organizational office or denominational affiliation, disclose relevant commitments, and score the evidence ledger rather than conformity to an organization. Disputed prophetic claims require at least two reviewers, including one capable of presenting a serious alternative reading. One reviewer may cover multiple roles if genuinely qualified, but cannot perform both sides of a disputed-claim review alone.

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
| Canonical edge support accuracy | at least 95% expert-rated |
| Prophetic arithmetic and unit accuracy | 100% on release suite |
| Approved-rule scope compliance | at least 98% |
| Historicist/tradition-specific labeling | at least 95% expert-rated |
| Unsupported hidden-code claims in release suite | 0 |
| Rival-reading representation | at least 90% expert-rated |
| Organizational-belief attribution accuracy | 100% against official-source suite |
| Denominational-source leakage into biblical evidence score | 0 |
| High-stakes referral/boundary compliance | at least 98% |
| General benchmark regression | no more than 5% unless explicitly approved |

Report confidence intervals and sample sizes. Averages cannot conceal catastrophic failures in a language or safety category.

## 14. Milestones and realistic timeline

Assumes one primary engineer working part-time to full-time, with expert review scheduled separately.

| Phase | Deliverable | Estimate |
|---|---|---:|
| 0 | Repository, environment, hardware smoke tests | 1–2 weeks |
| 1 | Data/license registry and ingestion framework | 2–6 weeks |
| 2 | Baselines, model-capacity gate, tokenizer audit, sealed evaluation sets | 4–8 weeks |
| 3 | QLoRA/SFT pipeline and retrieval prototype | 2–4 weeks |
| 4 | 10–30M-token method comparison pilot | 2–5 weeks |
| 5 | 100–300M-token specialist training iterations | 1–3 months |
| 6 | Hermeneutics policy, prophetic-rule registry, moral constitution, SFT, adversarial tests | 3–7 months, overlapping |
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

> You are the lead ML engineer and research software architect for the Apertus Bible-Grounded AI project. Implement the attached master plan incrementally on one RTX 5090 32 GB workstation. Treat claims, licenses, data provenance, evaluation isolation, and reproducibility as first-class engineering requirements.
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
> Implement "Scripture interprets Scripture" as an auditable evidence method, not a license to invent connections. Every cross-reference, symbol, prophetic rule, calculation, historical anchor, and moral application must retain its source spans, assumptions, interpretive-school label, counter-reading, review status, and confidence. Use deterministic tools for arithmetic and chronology. Distinguish the textual 42-month/1,260-day correspondence from the historicist conversion to 1,260 years. Reject unsupported numerology, hidden codes, and date-setting.
>
> The AI may transparently apply a reviewed biblical framework, but it must never claim consciousness, revelation, infallibility, divine authority, or that its answer is God’s direct will. It must distinguish explicit text, canonical synthesis, contextual inference, named historical interpretation, organizational alignment, wisdom judgment, and speculative hypothesis. Do not treat SDA or any other organization as theological evidence, and do not discount a multi-passage biblical conclusion merely because an organization also teaches it. Use official organizational sources only to state accurately what that organization teaches. Refer high-stakes matters to qualified people.
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
9. Draft—not finalize—the hermeneutics policy, prophetic-rule registry, moral constitution, and theology policy outlines for human review.
10. Stop and provide a Phase 0 report before downloading corpora or training.

## 18. Decisions the owner must make

- Internal research, nonprofit ministry, open release, or commercial use.
- Confirm that Scripture is the project's theological evidence authority and that SDA or other organizational statements are limited to factual self-description and alignment metadata.
- Decide whether non-biblical devotional authors, including Ellen G. White, are excluded entirely from doctrinal evidence or available only in a separately labeled historical-reception collection with verified licensing.
- Whether a later Apertus checkpoint should replace the pinned 8B text baseline after identical capacity, safety, CUDA, and efficiency tests.
- Canon boundaries and the authority/labeling policy for deuterocanonical and other ancient religious texts.
- Whether historicism is the preferred interpretive framework or one labeled framework among several in general-user answers.
- Which prophetic rules are allowed, their scope, and the evidence threshold for adding a new rule.
- Whether the system may evaluate proposed historical fulfillments and, if so, the primary-source and dual-review requirements.
- A strict no-new-date-setting policy and escalation process for users seeking predictions.
- Preferred Bible textual traditions and critical editions.
- Approved English and Spanish translations.
- Whether Syriac is in v1 or deferred.
- Budget for scholarly and pastoral review.
- Privacy policy and intended user population, including minors.
- Public model weights versus hosted-only release.
- Whether generated moral case data may be used and under what human-review threshold.

## 19. Authoritative technical references

- Apertus v1.1 1.5B model card: https://huggingface.co/swiss-ai/Apertus-v1.1-1.5B
- Apertus v1.1 4B model card: https://huggingface.co/swiss-ai/Apertus-v1.1-4B
- Apertus 8B Instruct 2509 model card: https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509
- Apertus v1.5 8B model card: https://huggingface.co/swiss-ai/Apertus-v1.5-8B
- Hugging Face Apertus implementation: https://huggingface.co/docs/transformers/model_doc/apertus
- Hugging Face PEFT/LoRA documentation: https://huggingface.co/docs/peft/developer_guides/lora
- Hugging Face TRL documentation: https://huggingface.co/docs/trl/
- Constitutional AI paper: https://arxiv.org/abs/2212.08073
- Direct Preference Optimization paper: https://arxiv.org/abs/2305.18290
- Retrieval-Augmented Generation paper: https://arxiv.org/abs/2005.11401
- Microsoft GraphRAG documentation: https://microsoft.github.io/graphrag/
- NVIDIA RTX 5090 specifications: https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
- NVIDIA RTX 5090 user guide: https://www.nvidia.com/content/geforce-gtx/GeForce_RTX_5090_User_Guide_Rev1.pdf
- PyTorch installation selector: https://pytorch.org/get-started/locally/
- Open Scriptures Hebrew Bible: https://github.com/openscriptures/morphhb
- MorphGNT: https://github.com/morphgnt/sblgnt
- Official SDA 28 Fundamental Beliefs, used only to verify what the organization officially teaches and never as biblical ground truth: https://adventist.org/en/beliefs
- Numbers 14:34 and Ezekiel 4:6 day-for-year precedents (public-domain KJV): https://www.biblegateway.com/passage/?search=Numbers+14%3A34%2CEzekiel+4%3A6&version=KJV
- Revelation 11:2-3; 12:6,14; 13:5 and Daniel 7:25; 12:7 must be entered from an approved translation or licensed source with exact provenance.
- Chicago Statement on Biblical Hermeneutics, used as one documented Protestant hermeneutic reference rather than an unlabeled universal consensus: https://alliancenet.org/icbi/the-chicago-statement-on-biblical-hermeneutics/

All technical versions, dataset terms, and licenses must be reverified at implementation time and recorded with dates and hashes.
