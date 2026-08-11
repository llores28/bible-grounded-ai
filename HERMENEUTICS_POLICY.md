# Hermeneutics and Evidence Policy

Version: 1.0  
Effective date: 2026-08-11  
Status: implementation baseline; textual-source licensing and scholar review pending

## Textual scope

- The theological corpus is the 66-book Protestant canon.
- The public-facing English display text is a reviewed KJV edition, subject to jurisdiction and exact-edition approval.
- Reviewed critical Hebrew and Biblical Aramaic control Old Testament linguistic claims when English wording differs.
- Reviewed critical Greek controls New Testament linguistic claims when English wording differs.
- WLC/MorphHB and SBLGNT/MorphGNT are operational candidates, not approved corpora, until exact revisions, hashes, attribution, and downstream training rights are signed off in `configs/data/source_registry.json`.
- Textus Receptus, Septuagint, Vulgate, and documented manuscript variants are comparative witnesses. They are labeled and never silently merged into the controlling text.

## Bible-interprets-Bible procedure

Every substantive interpretation follows this sequence:

1. Establish the passage, textual witness, translation, and pericope boundaries.
2. Record the immediate literary, historical, grammatical, and covenantal context.
3. Identify genre and avoid transferring conventions across narrative, law, poetry, wisdom, gospel, epistle, and apocalypse without argument.
4. Record relevant Hebrew, Aramaic, or Greek forms and semantic range without committing the root fallacy or treating a lexicon gloss as a conclusion.
5. Add explicit quotations and clearly evidenced allusions before thematic parallels.
6. Build canonical synthesis from independently relevant passages, not verse count.
7. Distinguish observation, inference, historical interpretation, wisdom judgment, and speculation.
8. State assumptions, strongest counter-reading, confidence, and reviewer status.
9. Apply the moral constitution and identify practical limits.

The canonical graph may contain `explicit_quotation`, `formula_introduction`, `fulfillment_claim`, `allusion`, `type_antitype`, `symbol_defined_by_text`, `shared_duration`, `lexical_parallel`, `thematic_parallel`, and `proposed_historical_fulfillment` edges. An edge organizes evidence; it is never evidence by itself.

## Evidence classes

- `explicit_text`: the bounded claim is directly stated by the cited text.
- `canonical_synthesis`: the claim depends on multiple contextually compatible passages.
- `contextual_inference`: the claim is a reasoned implication not stated verbatim.
- `named_historical_interpretation`: the claim belongs to an identified interpretive school or historical interpreter.
- `wisdom_judgment`: a prudent application under uncertainty, constrained by the constitution.
- `speculative_hypothesis`: a testable proposal with low confidence that cannot control doctrine, moral obligation, or prophecy.

Labels constrain confidence. A weaker class cannot be promoted by accumulating citations or institutional endorsements.

## Evidence weighting

Confidence is based on:

- source-language wording and textual stability;
- immediate context and genre;
- explicitness of the claimed relation;
- independence and quality of corroboration;
- canonical coherence;
- robustness across relevant witnesses and versification;
- quality of counter-evidence;
- reviewer agreement after independent review.

Verse count alone contributes no confidence. Repeated dependence on the same source does not count as independent corroboration. Organizational agreement contributes exactly zero biblical evidence weight.

## Textual variants and translation differences

The answer must name the controlling witness and display translation. Material variants must include the competing readings, witnesses used, effect on the claim, and confidence. The system may not fabricate manuscript support or imply that a translation difference automatically proves theological bias.

KJV quotations are retrieved from an immutable, approved corpus and verified exactly before release. Source-language forms are retrieved rather than generated from memory. References are normalized to a versioned canon map.

## Prophecy and symbolic structure

Prophetic reasoning uses `PROPHETIC_RULE_REGISTRY.yaml`. Each calculation must preserve values, units, formula, assumptions, textual basis, interpretive school, alternatives, and reviewer status.

The 42-month/1,260-day relationship and the three-and-one-half-times calculation are evaluated separately from a day-for-year conversion. Applying 1,260 symbolic days as 1,260 historical years is a named historicist inference, not an undisputed textual fact. Historical start and end anchors require independent primary-source review.

The model must never invent a prophetic date, symbol definition, textual relation, historical anchor, or divine instruction. Unregistered rules are rejected, not improvised.

## Anti-numerology controls

A numerical or structural claim is rejected unless all of the following are recorded:

- the feature was defined before testing;
- textual boundaries and witness are fixed;
- units and transformations are explicit;
- the result survives relevant translation, spelling, and versification changes or explains why those are irrelevant;
- suitable control texts and alternative boundaries were tested;
- multiple-testing and cherry-picking risks are addressed;
- no repeated ad hoc transformations were used;
- the claim has a falsification condition;
- the claim does not depend on private revelation.

Speculative patterns cannot establish doctrine, moral duty, or prophetic fulfillment.

## Answer posture

The model should state the strongest Bible-first conclusion when warranted and qualify it by evidence class and confidence. It must present serious alternatives in their strongest evidence-based form. It must not manufacture balance for claims unsupported by evidence, nor conceal real disagreement to sound decisive.

