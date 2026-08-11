"""Stable rendering of structured moral answers for SFT and user display."""

from __future__ import annotations

from .schemas import MoralAnswer


def _section(title: str, values: tuple[str, ...]) -> list[str]:
    lines = [f"## {title}"]
    lines.extend(f"- {value}" for value in values)
    return lines


def render_moral_answer(answer: MoralAnswer) -> str:
    """Render all required answer sections without exposing hidden reasoning traces."""

    lines: list[str] = []
    lines.extend(_section("Known facts", answer.known_facts))
    lines.extend(
        _section("Missing information", answer.missing_information or ("None identified.",))
    )
    lines.append("## Relevant commandments")
    for assessment in sorted(answer.commandment_assessments, key=lambda item: item.commandment):
        lines.append(
            f"- {assessment.commandment}: {assessment.verdict.value}. {assessment.rationale}"
        )
    lines.append("## Biblical evidence")
    for item in answer.evidence:
        quotation = f' "{item.quotation}"' if item.quotation else ""
        lines.append(
            f"- [{item.evidence_class.value}] {item.source_id} {item.reference}:{quotation} "
            f"{item.claim} Context: {item.immediate_context or 'Not supplied.'} "
            f"Language: {item.language_notes or 'No material language note.'}"
        )
    lines.extend(_section("Moral duties", answer.moral_duties))
    lines.extend(_section("Affected people", answer.affected_people))
    lines.extend(_section("Potential harm", answer.potential_harms))
    lines.extend(
        ("## Bible-first conclusion", f"{answer.conclusion} Confidence: {answer.confidence.value}.")
    )
    lines.extend(_section("Serious alternatives", answer.alternatives))
    lines.extend(_section("Safe practical options", answer.practical_options))
    lines.extend(
        _section("Human referral", answer.human_referral or ("None required for this case.",))
    )
    if answer.organizational_alignment:
        lines.append("## Optional organizational alignment")
        for item in answer.organizational_alignment:
            lines.append(
                f"- {item.organization}, {item.official_document}: {item.alignment}. "
                f"Biblical evidence weight: {item.evidence_weight:.1f}. Source: {item.source_url}"
            )
    return "\n".join(lines)


def build_sft_messages(answer: MoralAnswer, *, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": answer.request_text},
        {"role": "assistant", "content": render_moral_answer(answer)},
    ]
