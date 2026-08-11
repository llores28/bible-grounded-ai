"""Validate blinded pilot reviews and fail-closed adjudication records."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .dataset import read_jsonl
from .evidence_store import file_sha256
from .registry import load_json


@dataclass(frozen=True, slots=True)
class ReviewLedgerIssue:
    item_id: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ReviewLedgerReport:
    passed: bool
    assignment_count: int
    completed_review_count: int
    consensus_approved_count: int
    revision_required_count: int
    rejected_count: int
    incomplete_count: int
    issues: tuple[ReviewLedgerIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReviewLedgerValidator:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def validate(
        self,
        *,
        packets_path: str | Path = "data/pilot/candidate_review_packets.jsonl",
        assignments_path: str | Path = "data/pilot/reviewer_assignments.json",
        reviews_path: str | Path = "data/pilot/reviews.jsonl",
        adjudications_path: str | Path = "data/pilot/adjudications.jsonl",
    ) -> ReviewLedgerReport:
        issues: list[ReviewLedgerIssue] = []
        packet_file = self.root / packets_path
        assignment_file = self.root / assignments_path
        review_file = self.root / reviews_path
        adjudication_file = self.root / adjudications_path
        for name, path in (
            ("review packets", packet_file),
            ("reviewer assignments", assignment_file),
            ("review ledger", review_file),
        ):
            if not path.is_file():
                issues.append(
                    ReviewLedgerIssue("LEDGER", "LEDGER_INPUT_MISSING", f"{name} missing: {path}")
                )
        if issues:
            return ReviewLedgerReport(False, 0, 0, 0, 0, 0, 0, tuple(issues))

        packets = read_jsonl(packet_file)
        packet_by_id = {str(item.get("item_id", "")): item for item in packets}
        if len(packet_by_id) != len(packets):
            issues.append(
                ReviewLedgerIssue("LEDGER", "DUPLICATE_PACKET_ID", "packet item IDs must be unique")
            )
        for item_id, packet in packet_by_id.items():
            candidate_digest = str(packet.get("candidate_record_sha256", ""))
            if not re.fullmatch(r"[a-f0-9]{64}", candidate_digest):
                issues.append(
                    ReviewLedgerIssue(
                        item_id,
                        "CANDIDATE_RECORD_DIGEST_MISSING",
                        "review packets must contain a validated candidate record digest",
                    )
                )
            else:
                candidate_canonical = json.dumps(
                    packet.get("candidate_record"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if hashlib.sha256(candidate_canonical.encode("utf-8")).hexdigest() != candidate_digest:
                    issues.append(
                        ReviewLedgerIssue(
                            item_id,
                            "CANDIDATE_RECORD_DIGEST_MISMATCH",
                            "candidate record does not match candidate_record_sha256",
                        )
                    )
            claimed = str(packet.get("packet_sha256", ""))
            unsigned = {key: value for key, value in packet.items() if key != "packet_sha256"}
            canonical = json.dumps(
                unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if claimed != actual:
                issues.append(
                    ReviewLedgerIssue(
                        item_id,
                        "PACKET_DIGEST_MISMATCH",
                        "review packet contents do not match packet_sha256",
                    )
                )

        assignments_payload = load_json(assignment_file)
        if assignments_payload.get("candidate_packets_sha256") != file_sha256(packet_file):
            issues.append(
                ReviewLedgerIssue(
                    "LEDGER",
                    "ASSIGNMENT_PACKET_FILE_DIGEST_MISMATCH",
                    "assignments were not created for the current candidate packet file",
                )
            )
        if assignments_payload.get("reviewer_registry_sha256") != file_sha256(
            self.root / "configs/reviewers.json"
        ):
            issues.append(
                ReviewLedgerIssue(
                    "LEDGER",
                    "ASSIGNMENT_REVIEWER_DIGEST_MISMATCH",
                    "assignments were not created for the current reviewer registry",
                )
            )
        assignments = assignments_payload.get("assignments", [])
        if not isinstance(assignments, list):
            issues.append(
                ReviewLedgerIssue("LEDGER", "ASSIGNMENTS_INVALID", "assignments must be a list")
            )
            assignments = []
        assignment_by_id = {str(item.get("item_id", "")): item for item in assignments}
        if len(assignment_by_id) != len(assignments):
            issues.append(
                ReviewLedgerIssue(
                    "LEDGER", "DUPLICATE_ASSIGNMENT_ID", "assignment item IDs must be unique"
                )
            )
        if set(assignment_by_id) != set(packet_by_id):
            issues.append(
                ReviewLedgerIssue(
                    "LEDGER",
                    "ASSIGNMENT_PACKET_COVERAGE_MISMATCH",
                    "every packet must have exactly one assignment",
                )
            )
        for item_id, assignment in assignment_by_id.items():
            packet = packet_by_id.get(item_id)
            if packet is not None and assignment.get("packet_sha256") != packet.get(
                "packet_sha256"
            ):
                issues.append(
                    ReviewLedgerIssue(
                        item_id,
                        "ASSIGNMENT_PACKET_DIGEST_MISMATCH",
                        "assignment does not bind to the current candidate packet",
                    )
                )

        reviewers_payload = load_json(self.root / "configs/reviewers.json")
        active_reviewers = {
            str(item.get("reviewer_id")): item
            for item in reviewers_payload.get("reviewers", [])
            if item.get("status") == "active"
            and item.get("affiliations_disclosed") is True
            and item.get("independence_attested_on")
        }
        reviews = read_jsonl(review_file)
        reviews_by_item: dict[str, list[dict[str, Any]]] = {}
        seen_review_ids: set[str] = set()
        seen_item_reviewer: set[tuple[str, str]] = set()
        for review in reviews:
            item_id = str(review.get("item_id", ""))
            reviewer_id = str(review.get("reviewer_id", ""))
            review_id = str(review.get("review_id", ""))
            if review_id in seen_review_ids:
                issues.append(
                    ReviewLedgerIssue(item_id, "DUPLICATE_REVIEW_ID", f"duplicate {review_id}")
                )
            seen_review_ids.add(review_id)
            pair = (item_id, reviewer_id)
            if pair in seen_item_reviewer:
                issues.append(
                    ReviewLedgerIssue(
                        item_id,
                        "DUPLICATE_REVIEWER_DECISION",
                        f"reviewer submitted multiple decisions: {reviewer_id}",
                    )
                )
            seen_item_reviewer.add(pair)
            assignment = assignment_by_id.get(item_id)
            if assignment is None or reviewer_id not in assignment.get("reviewer_ids", []):
                issues.append(
                    ReviewLedgerIssue(
                        item_id,
                        "UNASSIGNED_REVIEWER",
                        f"reviewer is not assigned to this item: {reviewer_id}",
                    )
                )
            if assignment is not None and assignment.get("packet_sha256") != review.get(
                "packet_sha256"
            ):
                issues.append(
                    ReviewLedgerIssue(
                        item_id,
                        "ASSIGNMENT_PACKET_DIGEST_MISMATCH",
                        "review does not match the packet bound to its assignment",
                    )
                )
            if reviewer_id not in active_reviewers:
                issues.append(
                    ReviewLedgerIssue(
                        item_id,
                        "REVIEWER_NOT_ACTIVE",
                        f"reviewer is not active and attested: {reviewer_id}",
                    )
                )
            packet = packet_by_id.get(item_id)
            if packet is None or review.get("packet_sha256") != packet.get("packet_sha256"):
                issues.append(
                    ReviewLedgerIssue(
                        item_id,
                        "REVIEW_PACKET_DIGEST_MISMATCH",
                        "review does not bind to the current packet",
                    )
                )
            author_id = str(
                (packet or {}).get("candidate_record", {}).get("provenance", {}).get(
                    "author_id", ""
                )
            )
            if reviewer_id == author_id:
                issues.append(
                    ReviewLedgerIssue(
                        item_id,
                        "REVIEWER_IS_CANDIDATE_AUTHOR",
                        "a candidate author cannot count as an independent reviewer",
                    )
                )
            if review.get("decision") not in {"approve", "revise", "reject"}:
                issues.append(
                    ReviewLedgerIssue(item_id, "REVIEW_DECISION_INVALID", "invalid decision")
                )
            if not str(review.get("rationale", "")).strip():
                issues.append(
                    ReviewLedgerIssue(item_id, "REVIEW_RATIONALE_MISSING", "rationale required")
                )
            if review.get("independent_blind_attestation") is not True:
                issues.append(
                    ReviewLedgerIssue(
                        item_id,
                        "BLIND_REVIEW_NOT_ATTESTED",
                        "reviewer must attest to an independent blinded first pass",
                    )
                )
            if review.get("affiliations_disclosed") is not True:
                issues.append(
                    ReviewLedgerIssue(
                        item_id,
                        "REVIEW_AFFILIATION_NOT_DISCLOSED",
                        "affiliation disclosure is required",
                    )
                )
            if not self._valid_timestamp(review.get("reviewed_at")):
                issues.append(
                    ReviewLedgerIssue(
                        item_id, "REVIEW_TIMESTAMP_INVALID", "reviewed_at must include timezone"
                    )
                )
            reviews_by_item.setdefault(item_id, []).append(review)

        adjudications = read_jsonl(adjudication_file) if adjudication_file.is_file() else []
        adjudication_by_item: dict[str, list[dict[str, Any]]] = {}
        for adjudication in adjudications:
            adjudication_by_item.setdefault(str(adjudication.get("item_id", "")), []).append(
                adjudication
            )

        consensus_approved = 0
        revision_required = 0
        rejected = 0
        incomplete = 0
        completed_reviews = 0
        for item_id, assignment in assignment_by_id.items():
            assigned = set(assignment.get("reviewer_ids", []))
            item_reviews = reviews_by_item.get(item_id, [])
            submitted = {str(item.get("reviewer_id", "")) for item in item_reviews}
            if submitted != assigned:
                incomplete += 1
                issues.append(
                    ReviewLedgerIssue(
                        item_id,
                        "REVIEW_COVERAGE_INCOMPLETE",
                        f"submitted reviewers {sorted(submitted)}; assigned {sorted(assigned)}",
                    )
                )
                continue
            completed_reviews += len(item_reviews)
            decisions = {str(item.get("decision", "")) for item in item_reviews}
            if decisions == {"approve"}:
                consensus_approved += 1
                if item_id in adjudication_by_item:
                    issues.append(
                        ReviewLedgerIssue(
                            item_id,
                            "UNNECESSARY_ADJUDICATION",
                            "unanimous approval must not be overwritten by adjudication",
                        )
                    )
                continue
            item_adjudications = adjudication_by_item.get(item_id, [])
            if len(item_adjudications) != 1:
                incomplete += 1
                issues.append(
                    ReviewLedgerIssue(
                        item_id,
                        "ADJUDICATION_REQUIRED",
                        "non-unanimous reviews require exactly one adjudication",
                    )
                )
                continue
            adjudication = item_adjudications[0]
            adjudicator = str(adjudication.get("adjudicator_id", ""))
            if adjudicator in assigned or adjudicator not in active_reviewers:
                issues.append(
                    ReviewLedgerIssue(
                        item_id,
                        "ADJUDICATOR_NOT_INDEPENDENT",
                        "adjudicator must be a distinct active reviewer",
                    )
                )
            expected_review_ids = {str(item["review_id"]) for item in item_reviews}
            if set(adjudication.get("review_ids", [])) != expected_review_ids:
                issues.append(
                    ReviewLedgerIssue(
                        item_id,
                        "ADJUDICATION_REVIEW_SET_MISMATCH",
                        "adjudication must bind to every submitted review",
                    )
                )
            decision = adjudication.get("decision")
            if decision == "revise":
                revision_required += 1
            elif decision == "reject":
                rejected += 1
            else:
                issues.append(
                    ReviewLedgerIssue(
                        item_id,
                        "ADJUDICATION_CANNOT_ACCEPT_DISAGREEMENT",
                        "a changed candidate must receive fresh blinded reviews before acceptance",
                    )
                )
            if not str(adjudication.get("rationale", "")).strip():
                issues.append(
                    ReviewLedgerIssue(
                        item_id, "ADJUDICATION_RATIONALE_MISSING", "rationale required"
                    )
                )
            if not self._valid_timestamp(adjudication.get("adjudicated_at")):
                issues.append(
                    ReviewLedgerIssue(
                        item_id,
                        "ADJUDICATION_TIMESTAMP_INVALID",
                        "adjudicated_at must include timezone",
                    )
                )

        return ReviewLedgerReport(
            passed=not issues and consensus_approved == len(assignments),
            assignment_count=len(assignments),
            completed_review_count=completed_reviews,
            consensus_approved_count=consensus_approved,
            revision_required_count=revision_required,
            rejected_count=rejected,
            incomplete_count=incomplete,
            issues=tuple(issues),
        )

    @staticmethod
    def _valid_timestamp(value: object) -> bool:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return False
        return parsed.tzinfo is not None
