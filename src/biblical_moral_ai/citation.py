"""Exact, corpus-backed verification for biblical quotations and references."""

from __future__ import annotations

from collections.abc import Mapping

from .decisions import strongest_decision
from .schemas import (
    EvidenceClass,
    IssueSeverity,
    MoralAnswer,
    PipelineDecision,
    ReviewStatus,
    VerificationIssue,
    VerificationReport,
)


class CitationVerifier:
    """Check evidence against immutable source_id/reference/quotation mappings."""

    def __init__(
        self,
        corpora: Mapping[str, Mapping[str, str]],
        *,
        approved_source_ids: set[str] | None = None,
    ) -> None:
        self.corpora = {source: dict(entries) for source, entries in corpora.items()}
        self.approved_source_ids = approved_source_ids or set(self.corpora)

    def check(self, answer: MoralAnswer) -> VerificationReport:
        issues: list[VerificationIssue] = []
        seen_ids: set[str] = set()

        if not answer.evidence:
            issues.append(
                VerificationIssue(
                    code="CITATION_EVIDENCE_MISSING",
                    message="A biblical moral answer requires retrievable evidence.",
                    decision=PipelineDecision.CORRECT,
                    severity=IssueSeverity.CRITICAL,
                    field_path="evidence",
                )
            )

        for index, item in enumerate(answer.evidence):
            path = f"evidence[{index}]"
            if item.evidence_id in seen_ids:
                issues.append(
                    VerificationIssue(
                        code="CITATION_DUPLICATE_EVIDENCE_ID",
                        message=f"Duplicate evidence ID: {item.evidence_id}.",
                        decision=PipelineDecision.CORRECT,
                        field_path=f"{path}.evidence_id",
                    )
                )
            seen_ids.add(item.evidence_id)

            if item.source_id not in self.approved_source_ids:
                issues.append(
                    VerificationIssue(
                        code="CITATION_SOURCE_NOT_APPROVED",
                        message=f"Source is not approved for inference: {item.source_id}.",
                        decision=PipelineDecision.CORRECT,
                        severity=IssueSeverity.CRITICAL,
                        field_path=f"{path}.source_id",
                    )
                )
                continue

            corpus = self.corpora.get(item.source_id)
            if corpus is None:
                issues.append(
                    VerificationIssue(
                        code="CITATION_CORPUS_UNAVAILABLE",
                        message=f"Approved corpus is unavailable: {item.source_id}.",
                        decision=PipelineDecision.CORRECT,
                        severity=IssueSeverity.CRITICAL,
                        field_path=f"{path}.source_id",
                    )
                )
                continue

            expected = corpus.get(item.reference)
            if expected is None:
                issues.append(
                    VerificationIssue(
                        code="CITATION_REFERENCE_NOT_FOUND",
                        message=f"Reference does not exist in {item.source_id}: {item.reference}.",
                        decision=PipelineDecision.CORRECT,
                        severity=IssueSeverity.CRITICAL,
                        field_path=f"{path}.reference",
                    )
                )
                continue

            if item.evidence_class is EvidenceClass.EXPLICIT_TEXT and item.quotation is None:
                issues.append(
                    VerificationIssue(
                        code="CITATION_EXPLICIT_TEXT_QUOTE_MISSING",
                        message="Explicit-text evidence requires an exact retrieved quotation.",
                        decision=PipelineDecision.CORRECT,
                        field_path=f"{path}.quotation",
                    )
                )
            elif item.quotation is not None and item.quotation != expected:
                issues.append(
                    VerificationIssue(
                        code="CITATION_QUOTE_MISMATCH",
                        message=f"Quotation does not exactly match {item.source_id} {item.reference}.",
                        decision=PipelineDecision.CORRECT,
                        severity=IssueSeverity.CRITICAL,
                        field_path=f"{path}.quotation",
                    )
                )

            if item.reviewer_status not in {ReviewStatus.APPROVED, ReviewStatus.DISPUTED}:
                issues.append(
                    VerificationIssue(
                        code="CITATION_EVIDENCE_UNREVIEWED",
                        message=f"Evidence {item.evidence_id} is not approved or explicitly disputed.",
                        decision=PipelineDecision.CORRECT,
                        field_path=f"{path}.reviewer_status",
                    )
                )

        return VerificationReport(
            decision=strongest_decision(issues),
            issues=tuple(issues),
            checks={
                "evidence_present": bool(answer.evidence),
                "quotations_exact": not any(i.code == "CITATION_QUOTE_MISMATCH" for i in issues),
                "sources_approved": not any(
                    i.code == "CITATION_SOURCE_NOT_APPROVED" for i in issues
                ),
            },
        )
