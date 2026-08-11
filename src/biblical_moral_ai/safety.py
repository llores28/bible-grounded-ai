"""Pastoral-safety checks that fail closed on dangerous or authority-claiming advice."""

from __future__ import annotations

import re

from .decisions import strongest_decision
from .schemas import (
    IssueSeverity,
    MoralAnswer,
    PipelineDecision,
    VerificationIssue,
    VerificationReport,
)

_DANGEROUS_OUTPUTS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "SAFETY_CONCEAL_ABUSE",
        re.compile(
            r"(?:^|[.!?]\s+)(?:(?:do\s+not|don't|never)\s+report|(?:you\s+should\s+)?(?:hide|conceal|cover\s+up))\s+(?:the\s+)?(?:abuse|assault|crime)",
            re.IGNORECASE | re.MULTILINE,
        ),
        "The answer directs concealment of abuse, assault, or crime.",
    ),
    (
        "SAFETY_COERCIVE_SUBMISSION",
        re.compile(
            r"(?:^|[.!?]\s+)(?:you\s+)?(?:must|should)\s+(?:obey|submit\s+to|stay\s+with)\s+(?:an?\s+)?(?:abusive|violent|coercive)\s+(?:parent|spouse|pastor|authority)",
            re.IGNORECASE | re.MULTILINE,
        ),
        "The answer uses authority or submission to excuse abuse.",
    ),
    (
        "SAFETY_MEDICAL_REFUSAL",
        re.compile(
            r"(?:^|[.!?]\s+)(?:(?:you\s+should\s+)?(?:stop|skip|refuse|throw\s+away)\s+(?:your\s+)?(?:medication|medicine|treatment)|(?:prayer|faith)\s+(?:instead\s+of|means\s+you\s+do\s+not\s+need)\s+(?:a\s+)?(?:doctor|treatment|medicine))",
            re.IGNORECASE | re.MULTILINE,
        ),
        "The answer recommends medical refusal or replacing care with a spiritual claim.",
    ),
    (
        "SAFETY_FINANCIAL_EXPLOITATION",
        re.compile(
            r"(?:^|[.!?]\s+)(?:you\s+should\s+)?(?:send|give|donate)\s+(?:all\s+)?(?:your\s+)?money\b.{0,80}\b(?:blessing|healing|salvation|miracle)\b",
            re.IGNORECASE | re.MULTILINE,
        ),
        "The answer promotes financial exploitation through a promised spiritual benefit.",
    ),
    (
        "SAFETY_DIVINE_AUTHORITY_CLAIM",
        re.compile(
            r"(?:^|[.!?]\s+)(?:god\s+(?:told|commands|revealed\s+to)\s+(?:me|this\s+model)|i\s+speak\s+for\s+god|this\s+is\s+a\s+new\s+revelation)\b",
            re.IGNORECASE | re.MULTILINE,
        ),
        "The answer claims direct divine authority or new revelation.",
    ),
    (
        "SAFETY_LETHAL_AUTHORIZATION",
        re.compile(
            r"(?:^|[.!?]\s+)(?:you\s+(?:are\s+authorized\s+to|have\s+god's\s+permission\s+to)|god\s+(?:authorizes|commands)\s+you\s+to)\s+(?:kill|shoot|use\s+lethal\s+force)\b",
            re.IGNORECASE | re.MULTILINE,
        ),
        "The answer authorizes lethal action.",
    ),
)

_RISK_PATTERNS: dict[str, re.Pattern[str]] = {
    "abuse": re.compile(
        r"\b(?:abuse|abusive|assault|rape|molest|traffick|coercive|domestic\s+violence)\b",
        re.IGNORECASE,
    ),
    "self_harm": re.compile(
        r"\b(?:suicide|suicidal|self[- ]harm|kill\s+myself|end\s+my\s+life)\b", re.IGNORECASE
    ),
    "violence": re.compile(
        r"\b(?:murder|kill|shoot|stab|weapon|violent|self-defense|lethal\s+force)\b", re.IGNORECASE
    ),
    "religious_paranoia": re.compile(
        r"\b(?:god\s+told\s+me|direct\s+message\s+from\s+god|demons?\s+(?:watching|tracking|controlling)|religious\s+paranoia)\b",
        re.IGNORECASE,
    ),
    "scrupulosity": re.compile(
        r"\b(?:scrupulosity|unforgivable\s+sin|constant(?:ly)?\s+confess|every\s+thought\s+is\s+sin)\b",
        re.IGNORECASE,
    ),
    "medical": re.compile(
        r"\b(?:medication|medicine|doctor|medical\s+treatment|therapy|psychiatrist)\b",
        re.IGNORECASE,
    ),
    "financial_exploitation": re.compile(
        r"\b(?:prosperity\s+gospel|seed\s+offering|financial\s+exploitation|pay\s+for\s+(?:a\s+)?blessing)\b",
        re.IGNORECASE,
    ),
}

_IMMINENCE = re.compile(
    r"\b(?:right\s+now|now|imminent|today|tonight|has\s+a\s+weapon|about\s+to|immediately)\b",
    re.IGNORECASE,
)

_REFERRAL_TERMS: dict[str, tuple[str, ...]] = {
    "abuse": (
        "emergency",
        "law enforcement",
        "police",
        "safeguarding",
        "hotline",
        "protective services",
        "qualified professional",
    ),
    "self_harm": (
        "emergency",
        "crisis",
        "988",
        "hotline",
        "mental health professional",
        "trusted person",
    ),
    "violence": (
        "escape",
        "de-escalation",
        "emergency",
        "911",
        "law enforcement",
        "qualified legal",
        "safety plan",
    ),
    "religious_paranoia": (
        "mental health professional",
        "licensed clinician",
        "trusted pastor",
        "qualified professional",
    ),
    "scrupulosity": (
        "mental health professional",
        "licensed clinician",
        "trusted pastor",
        "qualified professional",
    ),
    "medical": ("doctor", "licensed clinician", "medical professional", "emergency"),
    "financial_exploitation": (
        "law enforcement",
        "financial professional",
        "trusted adviser",
        "report",
        "legal aid",
    ),
}


class PastoralSafetyChecker:
    def check(self, answer: MoralAnswer) -> VerificationReport:
        issues: list[VerificationIssue] = []
        output_text = " ".join((answer.conclusion, *answer.practical_options))
        full_text = " ".join(
            (answer.request_text, *answer.known_facts, output_text, *answer.human_referral)
        )

        for code, pattern, message in _DANGEROUS_OUTPUTS:
            if pattern.search(output_text):
                issues.append(
                    VerificationIssue(
                        code=code,
                        message=message,
                        decision=PipelineDecision.REFUSE,
                        severity=IssueSeverity.CRITICAL,
                        field_path="conclusion_or_practical_options",
                    )
                )

        referral_text = " ".join((*answer.practical_options, *answer.human_referral)).casefold()
        for category, pattern in _RISK_PATTERNS.items():
            if not pattern.search(full_text):
                continue
            required_terms = _REFERRAL_TERMS[category]
            if not any(term in referral_text for term in required_terms):
                issues.append(
                    VerificationIssue(
                        code=f"SAFETY_{category.upper()}_REFERRAL_MISSING",
                        message=f"A {category.replace('_', ' ')} case requires an appropriate human or emergency referral.",
                        decision=PipelineDecision.CORRECT,
                        severity=IssueSeverity.CRITICAL,
                        field_path="human_referral",
                    )
                )

        if _IMMINENCE.search(answer.request_text) and (
            _RISK_PATTERNS["self_harm"].search(answer.request_text)
            or _RISK_PATTERNS["violence"].search(answer.request_text)
            or _RISK_PATTERNS["abuse"].search(answer.request_text)
        ):
            issues.append(
                VerificationIssue(
                    code="SAFETY_IMMINENT_DANGER",
                    message="Imminent danger requires immediate human emergency escalation; the AI cannot resolve it autonomously.",
                    decision=PipelineDecision.ESCALATE,
                    severity=IssueSeverity.CRITICAL,
                    field_path="request_text",
                )
            )

        return VerificationReport(
            decision=strongest_decision(issues),
            issues=tuple(issues),
            checks={
                "dangerous_advice_absent": not any(
                    i.decision is PipelineDecision.REFUSE for i in issues
                ),
                "risk_referrals_present": not any(
                    i.code.endswith("REFERRAL_MISSING") for i in issues
                ),
            },
        )
