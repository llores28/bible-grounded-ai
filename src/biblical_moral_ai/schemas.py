"""Typed contracts shared by review, training, inference, and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Self


class EvidenceClass(StrEnum):
    EXPLICIT_TEXT = "explicit_text"
    CANONICAL_SYNTHESIS = "canonical_synthesis"
    CONTEXTUAL_INFERENCE = "contextual_inference"
    NAMED_HISTORICAL_INTERPRETATION = "named_historical_interpretation"
    WISDOM_JUDGMENT = "wisdom_judgment"
    SPECULATIVE_HYPOTHESIS = "speculative_hypothesis"


class Confidence(StrEnum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


class AssessmentVerdict(StrEnum):
    COMPLIANT = "compliant"
    VIOLATION = "violation"
    UNCERTAIN = "uncertain"
    NOT_APPLICABLE = "not_applicable"


class PipelineDecision(StrEnum):
    RELEASE = "release"
    CORRECT = "correct"
    ESCALATE = "escalate"
    REFUSE = "refuse"


class IssueSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ReviewStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DISPUTED = "disputed"


@dataclass(frozen=True, slots=True)
class CommandmentRule:
    number: int
    title: str
    scope: str
    hard_floor: bool
    applies_when: tuple[str, ...]
    requirements: tuple[str, ...]
    prohibitions: tuple[str, ...]
    qualifications: tuple[str, ...]
    anchor_passages: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 1 <= self.number <= 10:
            raise ValueError("commandment number must be between 1 and 10")
        if self.hard_floor != (self.number >= 5):
            raise ValueError(
                "only commandments 5-10 are configured as the interpersonal hard floor"
            )
        if self.scope not in {"duty_to_god", "interpersonal_floor"}:
            raise ValueError(f"unsupported commandment scope: {self.scope}")
        if not self.requirements or not self.prohibitions or not self.anchor_passages:
            raise ValueError(f"commandment {self.number} has an incomplete policy definition")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        return cls(
            number=int(value["number"]),
            title=str(value["title"]),
            scope=str(value["scope"]),
            hard_floor=bool(value["hard_floor"]),
            applies_when=tuple(value["applies_when"]),
            requirements=tuple(value["requirements"]),
            prohibitions=tuple(value["prohibitions"]),
            qualifications=tuple(value.get("qualifications", [])),
            anchor_passages=tuple(value["anchor_passages"]),
        )


@dataclass(frozen=True, slots=True)
class CommandmentAssessment:
    commandment: int
    verdict: AssessmentVerdict
    rationale: str
    evidence_ids: tuple[str, ...] = ()
    affected_people: tuple[str, ...] = ()
    remediation: str | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.commandment <= 10:
            raise ValueError("assessment commandment must be between 1 and 10")
        if not self.rationale.strip():
            raise ValueError("every commandment assessment requires a rationale")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        return cls(
            commandment=int(value["commandment"]),
            verdict=AssessmentVerdict(value["verdict"]),
            rationale=str(value["rationale"]),
            evidence_ids=tuple(value.get("evidence_ids", [])),
            affected_people=tuple(value.get("affected_people", [])),
            remediation=value.get("remediation"),
        )


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    evidence_id: str
    evidence_class: EvidenceClass
    source_id: str
    reference: str
    claim: str
    quotation: str | None = None
    immediate_context: str = ""
    language_notes: str = ""
    assumptions: tuple[str, ...] = ()
    reviewer_status: ReviewStatus = ReviewStatus.UNREVIEWED
    confidence: Confidence = Confidence.LOW

    def __post_init__(self) -> None:
        if not self.evidence_id.strip() or not self.source_id.strip() or not self.reference.strip():
            raise ValueError("evidence_id, source_id, and reference are required")
        if not self.claim.strip():
            raise ValueError("every evidence item requires a bounded claim")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        return cls(
            evidence_id=str(value["evidence_id"]),
            evidence_class=EvidenceClass(value["evidence_class"]),
            source_id=str(value["source_id"]),
            reference=str(value["reference"]),
            claim=str(value["claim"]),
            quotation=value.get("quotation"),
            immediate_context=str(value.get("immediate_context", "")),
            language_notes=str(value.get("language_notes", "")),
            assumptions=tuple(value.get("assumptions", [])),
            reviewer_status=ReviewStatus(value.get("reviewer_status", "unreviewed")),
            confidence=Confidence(value.get("confidence", "low")),
        )


@dataclass(frozen=True, slots=True)
class OrganizationalAlignment:
    organization: str
    official_document: str
    statement: str
    alignment: str
    source_url: str
    evidence_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.evidence_weight != 0.0:
            raise ValueError("organizational alignment must have zero biblical evidence weight")
        if not all(
            item.strip()
            for item in (
                self.organization,
                self.official_document,
                self.statement,
                self.alignment,
                self.source_url,
            )
        ):
            raise ValueError("organizational alignment fields cannot be blank")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        return cls(
            organization=str(value["organization"]),
            official_document=str(value["official_document"]),
            statement=str(value["statement"]),
            alignment=str(value["alignment"]),
            source_url=str(value["source_url"]),
            evidence_weight=float(value.get("evidence_weight", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class MoralAnswer:
    request_text: str
    known_facts: tuple[str, ...]
    missing_information: tuple[str, ...]
    commandment_assessments: tuple[CommandmentAssessment, ...]
    evidence: tuple[EvidenceItem, ...]
    moral_duties: tuple[str, ...]
    affected_people: tuple[str, ...]
    potential_harms: tuple[str, ...]
    conclusion: str
    confidence: Confidence
    alternatives: tuple[str, ...]
    practical_options: tuple[str, ...]
    human_referral: tuple[str, ...] = ()
    organizational_alignment: tuple[OrganizationalAlignment, ...] = ()
    answer_id: str = ""

    def __post_init__(self) -> None:
        if not self.request_text.strip() or not self.conclusion.strip():
            raise ValueError("request_text and conclusion are required")
        if not self.known_facts:
            raise ValueError("known_facts must contain at least one bounded fact")
        if not self.commandment_assessments:
            raise ValueError("at least one commandment assessment is required")
        if not self.moral_duties or not self.affected_people or not self.potential_harms:
            raise ValueError("moral duties, affected people, and potential harms are required")
        if not self.alternatives or not self.practical_options:
            raise ValueError("alternatives and practical options are required")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        return cls(
            request_text=str(value["request_text"]),
            known_facts=tuple(value["known_facts"]),
            missing_information=tuple(value.get("missing_information", [])),
            commandment_assessments=tuple(
                CommandmentAssessment.from_dict(item) for item in value["commandment_assessments"]
            ),
            evidence=tuple(EvidenceItem.from_dict(item) for item in value.get("evidence", [])),
            moral_duties=tuple(value["moral_duties"]),
            affected_people=tuple(value["affected_people"]),
            potential_harms=tuple(value["potential_harms"]),
            conclusion=str(value["conclusion"]),
            confidence=Confidence(value["confidence"]),
            alternatives=tuple(value["alternatives"]),
            practical_options=tuple(value["practical_options"]),
            human_referral=tuple(value.get("human_referral", [])),
            organizational_alignment=tuple(
                OrganizationalAlignment.from_dict(item)
                for item in value.get("organizational_alignment", [])
            ),
            answer_id=str(value.get("answer_id", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerificationIssue:
    code: str
    message: str
    decision: PipelineDecision
    severity: IssueSeverity = IssueSeverity.WARNING
    field_path: str = ""
    commandment: int | None = None


@dataclass(frozen=True, slots=True)
class VerificationReport:
    decision: PipelineDecision
    issues: tuple[VerificationIssue, ...] = ()
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.decision is PipelineDecision.RELEASE and not self.issues

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
