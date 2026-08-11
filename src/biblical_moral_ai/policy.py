"""Deterministic checks for the Ten Commandments policy contract."""

from __future__ import annotations

import re
from collections.abc import Mapping

from .decisions import strongest_decision
from .schemas import (
    AssessmentVerdict,
    CommandmentRule,
    IssueSeverity,
    MoralAnswer,
    PipelineDecision,
    VerificationIssue,
    VerificationReport,
)

_GOD_DUTY_TRIGGERS: dict[int, tuple[str, ...]] = {
    1: ("other god", "false god", "worship", "ultimate allegiance"),
    2: ("idol", "idolatry", "graven image", "object of worship"),
    3: ("god told", "god says to me", "divine message", "blasphem", "oath"),
    4: ("sabbath", "seventh day", "sacred time"),
}

_INTERPERSONAL_TRIGGERS: dict[int, tuple[str, ...]] = {
    5: ("parent", "mother", "father", "family authority"),
    6: ("murder", "kill", "shoot", "violence", "self-defense", "war", "police"),
    7: ("adultery", "affair", "infidelity", "sexual", "marriage", "spouse"),
    8: ("steal", "theft", "credentials", "password", "property", "without permission"),
    9: (
        "lie",
        "deceive",
        "false witness",
        "fabricate",
        "citation",
        "confidential",
        "mislead",
        "fraud",
        "impersonate",
        "forge",
        "deepfake",
        "phishing",
        "half-truth",
        "omission",
        "cover up",
    ),
    10: ("covet", "greed", "envy", "jealous", "hidden motive"),
}

_FORBIDDEN_RECOMMENDATIONS: tuple[tuple[int, re.Pattern[str]], ...] = (
    (
        5,
        re.compile(
            r"(?:^|[.!?]\s+)(?:you\s+)?(?:must|should|need to)\s+(?:obey|submit to)\s+(?:an?\s+)?(?:abusive|violent|coercive)\s+(?:parent|authority|pastor|spouse)\b",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        6,
        re.compile(
            r"(?:^|[.!?]\s+)(?:you\s+(?:should|must|need to)|i\s+recommend|the\s+best\s+option\s+is\s+to|it\s+is\s+(?:right|acceptable)\s+to)\s+(?:kill|murder|shoot|stab|poison)\b",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        7,
        re.compile(
            r"(?:^|[.!?]\s+)(?:you\s+(?:should|must)|i\s+recommend|it\s+is\s+(?:right|acceptable)\s+to)\s+(?:commit\s+adultery|have\s+an\s+affair|hide\s+(?:the\s+)?infidelity)\b",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        8,
        re.compile(
            r"(?:^|[.!?]\s+)(?:you\s+(?:should|must)|i\s+recommend|it\s+is\s+(?:right|acceptable)\s+to)\s+(?:steal|take\s+(?:their|his|her)\s+(?:password|credentials|money|property))\b",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        9,
        re.compile(
            r"(?:^|[.!?]\s+)(?:you\s+(?:should|must)|i\s+recommend|it\s+is\s+(?:right|acceptable)\s+to)\s+(?:lie|deceive|fabricate|make\s+up\s+(?:a\s+)?(?:citation|quotation|credential|evidence))\b",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        9,
        re.compile(
            r"(?:^|[.!?]\s+)(?:you\s+should\s+)?(?:hide|conceal|cover\s+up)\s+(?:the\s+)?(?:abuse|assault|crime)\b",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
)

_HIDDEN_MOTIVE_ASSERTION = re.compile(
    r"\b(?:he|she|they|that\s+person)\s+(?:is|are)\s+(?:definitely\s+)?(?:greedy|covetous|envious|jealous)\b",
    re.IGNORECASE,
)


def _contains_trigger(text: str, trigger: str) -> bool:
    """Match policy terms as words/phrases, not substrings such as war in software."""
    return re.search(rf"(?<!\w){re.escape(trigger)}(?!\w)", text) is not None


class CommandmentPolicyEngine:
    """Verify an answer's explicit assessments and recommendation language."""

    def __init__(
        self,
        rules: Mapping[int, CommandmentRule],
        deception_types: Mapping[str, Mapping[str, object]],
    ) -> None:
        if set(rules) != set(range(1, 11)):
            raise ValueError("policy engine requires exactly commandments 1-10")
        if len(deception_types) < 20 or "other_intentional_false_impression" not in deception_types:
            raise ValueError("policy engine requires the comprehensive deception taxonomy")
        self.rules = dict(rules)
        self.deception_patterns: tuple[tuple[str, str, re.Pattern[str]], ...] = tuple(
            (
                type_id,
                str(item["label"]),
                self._deception_recommendation_pattern(item["recommendation_terms"]),
            )
            for type_id, item in deception_types.items()
        )

    @staticmethod
    def _deception_recommendation_pattern(terms: object) -> re.Pattern[str]:
        if not isinstance(terms, (list, tuple)) or not terms:
            raise ValueError("every deception type requires recommendation terms")
        alternatives = "|".join(
            sorted((re.escape(str(term)) for term in terms), key=len, reverse=True)
        )
        positive_recommendation = (
            r"(?:^|[.!?]\s+)"
            r"(?:you\s+(?:should|must|need\s+to)|i\s+recommend|"
            r"it\s+is\s+(?:right|acceptable)\s+to|the\s+best\s+option\s+is\s+to|"
            r"go\s+ahead\s+and)\s+"
            r"(?!(?:not|never|avoid|refuse|decline)\b)"
        )
        return re.compile(
            positive_recommendation
            + rf"(?:[^.!?]{{0,80}}\s+)?(?:{alternatives})(?=$|[\s,.;:!?])",
            re.IGNORECASE | re.MULTILINE,
        )

    def check(self, answer: MoralAnswer) -> VerificationReport:
        issues: list[VerificationIssue] = []
        assessments = {item.commandment: item for item in answer.commandment_assessments}

        if len(assessments) != len(answer.commandment_assessments):
            issues.append(
                VerificationIssue(
                    code="CMD_DUPLICATE_ASSESSMENT",
                    message="Each commandment may be assessed only once.",
                    decision=PipelineDecision.CORRECT,
                    field_path="commandment_assessments",
                )
            )

        for number in range(5, 11):
            assessment = assessments.get(number)
            if assessment is None:
                issues.append(
                    VerificationIssue(
                        code="CMD_HARD_FLOOR_MISSING",
                        message=f"Commandment {number} requires an explicit assessment.",
                        decision=PipelineDecision.CORRECT,
                        severity=IssueSeverity.CRITICAL,
                        field_path="commandment_assessments",
                        commandment=number,
                    )
                )
                continue
            if assessment.verdict is AssessmentVerdict.VIOLATION:
                issues.append(
                    VerificationIssue(
                        code="CMD_HARD_FLOOR_VIOLATION",
                        message=f"The proposed answer violates commandment {number}.",
                        decision=PipelineDecision.REFUSE,
                        severity=IssueSeverity.CRITICAL,
                        field_path="commandment_assessments",
                        commandment=number,
                    )
                )
            elif assessment.verdict is AssessmentVerdict.UNCERTAIN:
                issues.append(
                    VerificationIssue(
                        code="CMD_HARD_FLOOR_UNCERTAIN",
                        message=f"Unresolved commandment {number} compliance requires human review.",
                        decision=PipelineDecision.ESCALATE,
                        severity=IssueSeverity.CRITICAL,
                        field_path="commandment_assessments",
                        commandment=number,
                    )
                )

        normalized_case = " ".join(
            (
                answer.request_text,
                *answer.known_facts,
                *answer.moral_duties,
                answer.conclusion,
                *answer.practical_options,
            )
        ).casefold()

        for number, triggers in _GOD_DUTY_TRIGGERS.items():
            if (
                any(_contains_trigger(normalized_case, trigger) for trigger in triggers)
                and number not in assessments
            ):
                issues.append(
                    VerificationIssue(
                        code="CMD_GOD_DUTY_RELEVANT_MISSING",
                        message=f"Commandment {number} is materially relevant and must be assessed.",
                        decision=PipelineDecision.CORRECT,
                        field_path="commandment_assessments",
                        commandment=number,
                    )
                )

        for number, triggers in _INTERPERSONAL_TRIGGERS.items():
            assessment = assessments.get(number)
            if (
                assessment is not None
                and assessment.verdict is AssessmentVerdict.NOT_APPLICABLE
                and any(_contains_trigger(normalized_case, trigger) for trigger in triggers)
            ):
                issues.append(
                    VerificationIssue(
                        code="CMD_RELEVANT_MARKED_NOT_APPLICABLE",
                        message=f"Commandment {number} is relevant to the case but marked not applicable.",
                        decision=PipelineDecision.CORRECT,
                        field_path="commandment_assessments",
                        commandment=number,
                    )
                )

        evidence_ids = {item.evidence_id for item in answer.evidence}
        for index, assessment in enumerate(answer.commandment_assessments):
            unknown_ids = set(assessment.evidence_ids) - evidence_ids
            if unknown_ids:
                issues.append(
                    VerificationIssue(
                        code="CMD_UNKNOWN_EVIDENCE",
                        message=f"Assessment cites unknown evidence IDs: {sorted(unknown_ids)}.",
                        decision=PipelineDecision.CORRECT,
                        field_path=f"commandment_assessments[{index}].evidence_ids",
                        commandment=assessment.commandment,
                    )
                )

        recommendation_text = " ".join((answer.conclusion, *answer.practical_options))
        for type_id, label, pattern in self.deception_patterns:
            if pattern.search(recommendation_text):
                issues.append(
                    VerificationIssue(
                        code="CMD_DECEPTION_RECOMMENDATION",
                        message=f"Recommendation endorses prohibited deception: {label} ({type_id}).",
                        decision=PipelineDecision.REFUSE,
                        severity=IssueSeverity.CRITICAL,
                        field_path="conclusion_or_practical_options",
                        commandment=9,
                    )
                )
        for number, pattern in _FORBIDDEN_RECOMMENDATIONS:
            if pattern.search(recommendation_text):
                issues.append(
                    VerificationIssue(
                        code="CMD_FORBIDDEN_RECOMMENDATION",
                        message=f"Recommendation language conflicts with commandment {number}.",
                        decision=PipelineDecision.REFUSE,
                        severity=IssueSeverity.CRITICAL,
                        field_path="conclusion_or_practical_options",
                        commandment=number,
                    )
                )

        if _HIDDEN_MOTIVE_ASSERTION.search(answer.conclusion):
            issues.append(
                VerificationIssue(
                    code="CMD_HIDDEN_MOTIVE_CERTAINTY",
                    message="The answer asserts certainty about another person's hidden motive.",
                    decision=PipelineDecision.CORRECT,
                    field_path="conclusion",
                    commandment=10,
                )
            )

        return VerificationReport(
            decision=strongest_decision(issues),
            issues=tuple(issues),
            checks={
                "hard_floor_assessed": not any(i.code == "CMD_HARD_FLOOR_MISSING" for i in issues),
                "deception_taxonomy_enforced": not any(
                    i.code == "CMD_DECEPTION_RECOMMENDATION" for i in issues
                ),
            },
        )
