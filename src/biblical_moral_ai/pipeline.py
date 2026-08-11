"""Post-generation verification and release decision orchestration."""

from __future__ import annotations

from collections.abc import Iterable

from .citation import CitationVerifier
from .decisions import strongest_decision
from .policy import CommandmentPolicyEngine
from .safety import PastoralSafetyChecker
from .schemas import (
    Confidence,
    EvidenceClass,
    IssueSeverity,
    MoralAnswer,
    PipelineDecision,
    VerificationIssue,
    VerificationReport,
)


class InferenceReviewPipeline:
    """Compose independent verifiers and fail closed before answer delivery."""

    def __init__(
        self,
        *,
        commandment_policy: CommandmentPolicyEngine,
        citation_verifier: CitationVerifier,
        safety_checker: PastoralSafetyChecker | None = None,
        organizational_source_ids: Iterable[str] = (),
    ) -> None:
        self.commandment_policy = commandment_policy
        self.citation_verifier = citation_verifier
        self.safety_checker = safety_checker or PastoralSafetyChecker()
        self.organizational_source_ids = set(organizational_source_ids)

    def review(self, answer: MoralAnswer) -> VerificationReport:
        reports = (
            self.commandment_policy.check(answer),
            self.citation_verifier.check(answer),
            self.safety_checker.check(answer),
        )
        issues = [issue for report in reports for issue in report.issues]

        for index, item in enumerate(answer.evidence):
            if item.source_id in self.organizational_source_ids:
                issues.append(
                    VerificationIssue(
                        code="ORG_SOURCE_LEAKAGE",
                        message="An organizational source was placed in the biblical evidence ledger.",
                        decision=PipelineDecision.CORRECT,
                        severity=IssueSeverity.CRITICAL,
                        field_path=f"evidence[{index}].source_id",
                    )
                )

        evidence_classes = {item.evidence_class for item in answer.evidence}
        if (
            answer.confidence is Confidence.HIGH
            and evidence_classes
            and evidence_classes
            <= {
                EvidenceClass.SPECULATIVE_HYPOTHESIS,
                EvidenceClass.WISDOM_JUDGMENT,
            }
        ):
            issues.append(
                VerificationIssue(
                    code="CONFIDENCE_EXCEEDS_EVIDENCE",
                    message="High confidence cannot rest only on wisdom judgment or speculation.",
                    decision=PipelineDecision.CORRECT,
                    field_path="confidence",
                )
            )

        checks = {name: passed for report in reports for name, passed in report.checks.items()}
        checks["organizational_sources_segregated"] = not any(
            issue.code == "ORG_SOURCE_LEAKAGE" for issue in issues
        )
        checks["confidence_calibrated"] = not any(
            issue.code == "CONFIDENCE_EXCEEDS_EVIDENCE" for issue in issues
        )

        return VerificationReport(
            decision=strongest_decision(issues),
            issues=tuple(issues),
            checks=checks,
        )
