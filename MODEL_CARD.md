# Model Card: Biblical Moral AI

Version: 0.1  
Status: untrained candidate; release prohibited

## Model

- Proposed base: `swiss-ai/Apertus-8B-Instruct-2509`
- Base model license reported by the official model card: Apache-2.0
- Architecture: 8B decoder-only text model
- Adaptation: 4-bit NF4 QLoRA supervised fine-tuning, followed by independently reviewed DPO
- Retrieval: approved KJV and source-language corpora plus a reviewed canonical evidence graph
- Deterministic tools: citation verifier, commandment checker, pastoral-safety checker, and unit-preserving prophetic arithmetic

The base identifier and license were rechecked against the official model card on 2026-08-11. `swiss-ai/Apertus-v1.5-8B` is tracked as a future multimodal candidate, not the training baseline, because its current integration requires a project-specific Transformers branch and compatibility with PEFT, quantization, and the local CUDA target must be proven first.

## Intended users and uses

The primary audience is scholars and pastors evaluating Bible-grounded moral arguments. Intended uses include structured passage analysis, moral case comparison, preparation of review drafts, retrieval-backed study, and testing interpretive hypotheses.

The model is not intended to provide divine revelation, final church rulings, autonomous pastoral care, legal or medical decisions, mental-health diagnosis, financial authority, surveillance, weapon control, or authorization of force.

## Required output behavior

Every answer follows the `MoralAnswer` contract and is checked before delivery. It distinguishes facts from missing information, identifies commandments, cites evidence, records context and language limits, analyzes duties and harms, reaches a qualified conclusion, presents serious alternatives, and offers safe options and referral.

Commandments 5-10 are a hard interpersonal floor. Commandments 1-4 are assessed when relevant to duties toward God. Organizational alignment is optional metadata with zero biblical evidence weight.

## Training status

No model adapter has been trained. No expert-reviewed SFT or preference dataset has been accepted. Reviewed pilot, full training, and release commands must fail closed until source licensing, dataset counts, reviewer coverage, CUDA checks, and sealed-evaluation custody pass preflight. A separate explicitly acknowledged research configuration may run only a local two-step smoke test on labeled unreviewed candidates; its rows and run manifest are release-ineligible and cannot satisfy reviewed or release gates.

Configuration files in `configs/training/` specify the proposed experiments but are not benchmark results. Hyperparameters must be validated with a smoke test and recorded hardware/software manifest before a full run.

## Evaluation and release gates

Release requires exact KJV quotations and prophetic arithmetic, at least 99% citation accuracy, zero fabricated sources, zero organizational-source leakage into biblical evidence, zero accepted hard-floor violations, a 100% pass rate across the public deception taxonomy, advanced content-review, and truthful-confidentiality controls, complete dual review for high-impact cases, and the minimum reviewed dataset counts. Failed cases, reviewer methodology, known limitations, and reproducible metrics must be published.

Every generated structured answer also passes a fail-closed content-quality review for explicit contradictions, confidence and evidence-status mismatches, unsupported source-language conclusions, verdict-rationale conflicts, unresolved placeholders, vagueness, duplication, and action-sentence clarity. This layer detects reviewable defects; it does not prove doctrinal truth or replace qualified human review.

## Limitations

- An 8B model can still hallucinate, flatten textual nuance, imitate confidence, or fail under adversarial prompting.
- Retrieval quality and corpus errors can propagate into answers.
- Deterministic keyword checks are defense in depth, not semantic proof of morality.
- Biblical interpretation includes genuine disagreement that cannot be removed by training.
- KJV wording is not the controlling linguistic evidence when the reviewed source text differs.
- The system cannot know hidden motives, imminent facts, legal jurisdiction, or whether a user's account is complete.
- Safety filters can over-block legitimate scholarship and under-detect indirect harmful advice; high-impact use requires qualified human review.

## Base-model sources

- <https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509>
- <https://huggingface.co/swiss-ai/Apertus-v1.5-8B>

