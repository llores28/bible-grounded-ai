"""Validate fully authored pilot candidates before independent human review."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .citation import CitationVerifier
from .content_review import AdvancedContentReviewer
from .dataset import ReviewedDatasetValidator, read_jsonl
from .decisions import strongest_decision
from .evidence_store import EvidenceStore, file_sha256
from .pipeline import InferenceReviewPipeline
from .policy import CommandmentPolicyEngine
from .registry import (
    load_commandment_rules,
    load_content_review_rules,
    load_deception_taxonomy,
    load_json,
)
from .reviewers import ReviewerWorkflow, is_active_reviewer, reviewer_is_qualified
from .schemas import MoralAnswer, PipelineDecision


@dataclass(frozen=True, slots=True)
class CandidateIssue:
    item_id: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class CandidateAuditReport:
    passed: bool
    counts: dict[str, int]
    issues: tuple[CandidateIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CANDIDATE_PATHS = {
    "sft": "data/pilot/candidates/sft.jsonl",
    "preferences": "data/pilot/candidates/preferences.jsonl",
    "evals": "data/pilot/candidates/evals.jsonl",
}


class PilotCandidateWorkflow:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def audit(self) -> CandidateAuditReport:
        corpus_path = self.root / "data/index/citation_corpus.json"
        if not corpus_path.is_file():
            return CandidateAuditReport(
                False,
                {},
                (
                    CandidateIssue(
                        "CANDIDATES",
                        "CITATION_CORPUS_MISSING",
                        "run build-evidence before auditing candidates",
                    ),
                ),
            )
        queue = load_json(self.root / "configs/pilot/draft_scenarios.json")
        drafts = {
            split: {str(item["item_id"]): item for item in queue[split]}
            for split in _CANDIDATE_PATHS
        }
        registry = load_json(self.root / "configs/data/source_registry.json")
        sources = {
            str(item["source_id"]): item
            for item in registry.get("sources", [])
            if item.get("status") == "approved"
        }
        corpus_payload = load_json(corpus_path)
        pipeline = InferenceReviewPipeline(
            commandment_policy=CommandmentPolicyEngine(
                load_commandment_rules(self.root / "configs/commandments.json"),
                load_deception_taxonomy(self.root / "configs/deception_taxonomy.json"),
            ),
            content_reviewer=AdvancedContentReviewer(
                load_content_review_rules(self.root / "configs/content_review_rules.json")
            ),
            citation_verifier=CitationVerifier(
                corpus_payload.get("sources", corpus_payload)
            ),
            organizational_source_ids=corpus_payload.get("organizational_source_ids", []),
        )
        issues: list[CandidateIssue] = []
        counts: dict[str, int] = {}
        for split, relative_path in _CANDIDATE_PATHS.items():
            path = self.root / relative_path
            if not path.is_file():
                counts[split] = 0
                issues.append(
                    CandidateIssue(
                        split,
                        "CANDIDATE_FILE_MISSING",
                        f"authored candidate file is missing: {relative_path}",
                    )
                )
                continue
            try:
                envelopes = read_jsonl(path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                counts[split] = 0
                issues.append(CandidateIssue(split, "CANDIDATE_FILE_INVALID", str(exc)))
                continue
            counts[split] = len(envelopes)
            by_id = {str(item.get("item_id", "")): item for item in envelopes}
            if len(by_id) != len(envelopes):
                issues.append(
                    CandidateIssue(split, "DUPLICATE_CANDIDATE_ID", "item IDs must be unique")
                )
            missing = sorted(set(drafts[split]) - set(by_id))
            extra = sorted(set(by_id) - set(drafts[split]))
            if missing or extra:
                issues.append(
                    CandidateIssue(
                        split,
                        "CANDIDATE_COVERAGE_MISMATCH",
                        f"missing={missing}; extra={extra}",
                    )
                )
            seen_record_ids: set[str] = set()
            for item_id, envelope in by_id.items():
                draft = drafts[split].get(item_id)
                if draft is None:
                    continue
                record = envelope.get("record")
                if not isinstance(record, dict):
                    issues.append(
                        CandidateIssue(item_id, "CANDIDATE_RECORD_MISSING", "record is required")
                    )
                    continue
                if not isinstance(envelope.get("candidate_revision"), int) or int(
                    envelope.get("candidate_revision", 0)
                ) < 1:
                    issues.append(
                        CandidateIssue(
                            item_id,
                            "CANDIDATE_REVISION_INVALID",
                            "candidate_revision must be a positive integer",
                        )
                    )
                if record.get("status") != "candidate":
                    issues.append(
                        CandidateIssue(
                            item_id,
                            "CANDIDATE_STATUS_INVALID",
                            "record status must be candidate before review",
                        )
                    )
                if record.get("reviews") not in (None, []):
                    issues.append(
                        CandidateIssue(
                            item_id,
                            "PREPOPULATED_REVIEWS_FORBIDDEN",
                            "candidate records cannot contain reviews before blinded review",
                        )
                    )
                id_field, id_pattern = {
                    "sft": ("record_id", r"SFT-PILOT-[A-Z0-9_-]+"),
                    "preferences": ("pair_id", r"DPO-PILOT-[A-Z0-9_-]+"),
                    "evals": ("case_id", r"EVAL-PILOT-[A-Z0-9_-]+"),
                }[split]
                if not re.fullmatch(id_pattern, str(record.get(id_field, ""))):
                    issues.append(
                        CandidateIssue(
                            item_id,
                            "CANDIDATE_RECORD_ID_INVALID",
                            f"record.{id_field} must match {id_pattern}",
                        )
                    )
                record_id = str(record.get(id_field, ""))
                if record_id in seen_record_ids:
                    issues.append(
                        CandidateIssue(
                            item_id,
                            "DUPLICATE_CANDIDATE_RECORD_ID",
                            f"duplicate record identifier: {record_id}",
                        )
                    )
                seen_record_ids.add(record_id)
                for field in ("scenario_id", "high_impact"):
                    if record.get(field) != draft.get(field):
                        issues.append(
                            CandidateIssue(
                                item_id,
                                "CANDIDATE_DRAFT_MISMATCH",
                                f"record.{field} must match the curated draft",
                            )
                        )
                if self._category(record.get("category", "")) != self._category(
                    draft.get("category", "")
                ):
                    issues.append(
                        CandidateIssue(
                            item_id,
                            "CANDIDATE_DRAFT_MISMATCH",
                            "record.category must match the curated draft",
                        )
                    )
                if split in {"preferences", "evals"} and record.get(
                    "expected_decision", "release"
                ) != draft.get("expected_decision", "release"):
                    issues.append(
                        CandidateIssue(
                            item_id,
                            "CANDIDATE_DRAFT_MISMATCH",
                            "record.expected_decision must match the curated draft",
                        )
                    )
                issues.extend(
                    self._validate_provenance(item_id, record, draft, sources)
                )
                if split in {"sft", "evals"}:
                    issues.extend(
                        self._validate_answer_candidate(
                            item_id,
                            record,
                            draft,
                            pipeline,
                            expected_decision=(
                                PipelineDecision(
                                    str(record.get("expected_decision", "release"))
                                )
                                if split == "evals"
                                else PipelineDecision.RELEASE
                            ),
                        )
                    )
                else:
                    issues.extend(
                        self._validate_preference_candidate(item_id, record, draft, pipeline)
                    )
        return CandidateAuditReport(not issues, counts, tuple(issues))

    def build_review_packets(
        self,
        output_path: str | Path = "data/pilot/candidate_review_packets.jsonl",
    ) -> dict[str, Any]:
        report = self.audit()
        if not report.passed:
            raise ValueError(
                "candidate audit failed: "
                + " | ".join(f"{issue.code}: {issue.message}" for issue in report.issues[:10])
            )
        queue = load_json(self.root / "configs/pilot/draft_scenarios.json")
        drafts = {
            split: {str(item["item_id"]): item for item in queue[split]}
            for split in _CANDIDATE_PATHS
        }
        registry = load_json(self.root / "configs/data/source_registry.json")
        sources = {str(item["source_id"]): item for item in registry["sources"]}
        packets: list[dict[str, Any]] = []
        with EvidenceStore(self.root / "data/index/biblical_evidence.sqlite3") as store:
            for split, relative_path in _CANDIDATE_PATHS.items():
                for envelope in read_jsonl(self.root / relative_path):
                    item_id = str(envelope["item_id"])
                    draft = drafts[split][item_id]
                    passage = store.get_passage(draft["source_id"], draft["reference"])
                    if passage is None:
                        raise ValueError(f"evidence disappeared during packet build: {item_id}")
                    record_canonical = json.dumps(
                        envelope["record"],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    record_digest = hashlib.sha256(record_canonical.encode("utf-8")).hexdigest()
                    category = self._category(draft["category"])
                    packet = {
                        "item_id": item_id,
                        "split": split,
                        "candidate_revision": envelope["candidate_revision"],
                        "candidate_record": envelope["record"],
                        "candidate_record_sha256": record_digest,
                        "category": category,
                        "required_independent_reviewers": (
                            2
                            if split == "preferences"
                            or category in ReviewedDatasetValidator.SENSITIVE_CATEGORIES
                            else 1
                        ),
                        "evidence_snapshot": {
                            "source_id": passage.source_id,
                            "reference": passage.reference,
                            "quotation": passage.text,
                            "language": passage.language,
                            "source_revision": sources[passage.source_id]["revision"],
                            "canonical_source_sha256": sources[passage.source_id]["sha256"],
                            "license_decision_id": sources[passage.source_id]["approval"][
                                "decision_id"
                            ],
                        },
                    }
                    canonical = json.dumps(
                        packet, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    packet["packet_sha256"] = hashlib.sha256(
                        canonical.encode("utf-8")
                    ).hexdigest()
                    packets.append(packet)
        destination = self.root / output_path
        self._write_jsonl(destination, packets)
        return {
            "status": "candidate_review_packets_built",
            "output_path": str(destination),
            "packet_count": len(packets),
            "packet_file_sha256": file_sha256(destination),
        }

    def assign_reviewers(
        self,
        packets_path: str | Path = "data/pilot/candidate_review_packets.jsonl",
        output_path: str | Path = "data/pilot/reviewer_assignments.json",
    ) -> dict[str, Any]:
        packet_file = self.root / packets_path
        if not packet_file.is_file():
            raise ValueError("build validated candidate review packets before assigning reviewers")
        packets = read_jsonl(packet_file)
        registry = load_json(self.root / "configs/reviewers.json")
        readiness = ReviewerWorkflow(self.root).audit_readiness()
        if not readiness.passed:
            raise ValueError(
                "reviewer readiness failed: "
                + " | ".join(issue.message for issue in readiness.issues[:10])
            )
        allowed = {str(item) for item in registry.get("qualified_categories", [])}
        active = [
            item
            for item in registry.get("reviewers", [])
            if is_active_reviewer(item, allowed_qualifications=allowed)
        ]
        if len(active) < 2:
            raise ValueError(
                "at least two active reviewers with affiliation disclosure and "
                "independence attestations are required"
            )
        assignments: list[dict[str, Any]] = []
        for rotation, packet in enumerate(packets):
            category = str(packet["category"])
            required = int(packet["required_independent_reviewers"])
            author_id = str(
                packet.get("candidate_record", {}).get("provenance", {}).get("author_id", "")
            )
            qualified = [
                reviewer
                for reviewer in active
                if reviewer.get("reviewer_id") != author_id
                and reviewer_is_qualified(reviewer, category)
            ]
            if len(qualified) < required:
                raise ValueError(
                    f"not enough qualified reviewers for {packet['item_id']}: "
                    f"{len(qualified)}/{required}"
                )
            offset = rotation % len(qualified)
            selected = (qualified[offset:] + qualified[:offset])[:required]
            assignments.append(
                {
                    "item_id": packet["item_id"],
                    "packet_sha256": packet["packet_sha256"],
                    "reviewer_ids": [reviewer["reviewer_id"] for reviewer in selected],
                    "review_mode": "independent_blinded_first_pass",
                    "adjudication_required_on_disagreement": True,
                }
            )
        destination = self.root / output_path
        self._write_json(
            destination,
            {
                "schema_version": "1.0",
                "candidate_packets_sha256": file_sha256(packet_file),
                "reviewer_registry_sha256": file_sha256(
                    self.root / "configs/reviewers.json"
                ),
                "assignments": assignments,
            },
        )
        return {
            "status": "reviewers_assigned",
            "output_path": str(destination),
            "assignment_count": len(assignments),
        }

    def finalize_reviewed_pilot(
        self,
        *,
        reviews_path: str | Path = "data/pilot/reviews.jsonl",
    ) -> dict[str, Any]:
        from .review_ledger import ReviewLedgerValidator

        candidate_report = self.audit()
        if not candidate_report.passed:
            raise ValueError("candidate audit must pass before finalization")
        review_report = ReviewLedgerValidator(self.root).validate(reviews_path=reviews_path)
        if not review_report.passed:
            raise ValueError("review ledger must show unanimous completed approval")
        reviews = read_jsonl(self.root / reviews_path)
        reviews_by_item: dict[str, list[dict[str, Any]]] = {}
        for review in reviews:
            reviews_by_item.setdefault(str(review["item_id"]), []).append(review)

        accepted: dict[str, list[dict[str, Any]]] = {
            "sft": [],
            "preferences": [],
            "evals": [],
        }
        for split, relative_path in _CANDIDATE_PATHS.items():
            for envelope in read_jsonl(self.root / relative_path):
                item_id = str(envelope["item_id"])
                record = deepcopy(envelope["record"])
                record["status"] = "accepted"
                answer_fields = (
                    ("chosen", "rejected") if split == "preferences" else ("answer",)
                )
                for answer_field in answer_fields:
                    for evidence in record[answer_field].get("evidence", []):
                        evidence["reviewer_status"] = "approved"
                record["reviews"] = [
                    {
                        "reviewer_id": review["reviewer_id"],
                        "decision": (
                            "approve_chosen" if split == "preferences" else "approve"
                        ),
                        "rationale": review["rationale"],
                        "reviewed_at": review["reviewed_at"],
                        "affiliations_disclosed": True,
                        "independent": True,
                    }
                    for review in reviews_by_item[item_id]
                ]
                accepted[split].append(record)

        corpus_payload = load_json(self.root / "data/index/citation_corpus.json")
        pipeline = InferenceReviewPipeline(
            commandment_policy=CommandmentPolicyEngine(
                load_commandment_rules(self.root / "configs/commandments.json"),
                load_deception_taxonomy(self.root / "configs/deception_taxonomy.json"),
            ),
            content_reviewer=AdvancedContentReviewer(
                load_content_review_rules(self.root / "configs/content_review_rules.json")
            ),
            citation_verifier=CitationVerifier(
                corpus_payload.get("sources", corpus_payload)
            ),
            organizational_source_ids=corpus_payload.get("organizational_source_ids", []),
        )
        validator = ReviewedDatasetValidator(
            pipeline,
            source_registry=load_json(self.root / "configs/data/source_registry.json"),
            reviewer_registry=load_json(self.root / "configs/reviewers.json"),
        )
        validation = {
            "sft": validator.validate_sft(accepted["sft"]),
            "preferences": validator.validate_preferences(accepted["preferences"]),
            "evals": validator.validate_evals(accepted["evals"]),
        }
        failed = {
            split: report.issues for split, report in validation.items() if not report.passed
        }
        if failed:
            first = next(iter(failed.values()))[0]
            raise ValueError(f"final accepted record validation failed: {first.code}: {first.message}")

        manifest_path = self.root / "data/registry/pilot_manifest.json"
        manifest = load_json(manifest_path)
        split_map = {
            "sft": "sft_pilot",
            "preferences": "preference_pilot",
            "evals": "evaluation_pilot",
        }
        staged: list[tuple[Path, Path, str, int]] = []
        try:
            for split, manifest_split in split_map.items():
                destination = self.root / manifest["splits"][manifest_split]["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=destination.parent, delete=False
                ) as handle:
                    for record in accepted[split]:
                        handle.write(
                            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                        )
                    temporary = Path(handle.name)
                staged.append(
                    (temporary, destination, file_sha256(temporary), len(accepted[split]))
                )
            for temporary, destination, digest, count in staged:
                temporary.replace(destination)
                manifest_split = next(
                    key
                    for split, key in split_map.items()
                    if self.root / manifest["splits"][key]["path"] == destination
                )
                manifest["splits"][manifest_split]["accepted_count"] = count
                manifest["splits"][manifest_split]["sha256"] = digest
            manifest["status"] = "review_complete_pending_pilot_preflight"
            self._write_json(manifest_path, manifest)
        finally:
            for temporary, _, _, _ in staged:
                temporary.unlink(missing_ok=True)
        return {
            "status": "accepted_pilot_written",
            "counts": {split: len(records) for split, records in accepted.items()},
            "manifest_path": str(manifest_path),
        }

    def _validate_answer_candidate(
        self,
        item_id: str,
        record: dict[str, Any],
        draft: dict[str, Any],
        pipeline: InferenceReviewPipeline,
        expected_decision: PipelineDecision,
    ) -> list[CandidateIssue]:
        issues: list[CandidateIssue] = []
        try:
            answer = MoralAnswer.from_dict(record["answer"])
            if answer.request_text != draft["prompt"]:
                issues.append(
                    CandidateIssue(
                        item_id,
                        "CANDIDATE_PROMPT_MISMATCH",
                        "answer.request_text must exactly match the curated prompt",
                    )
                )
            report = pipeline.review(answer)
            if self._candidate_decision(report) is not expected_decision:
                issues.extend(
                    [
                        CandidateIssue(
                            item_id,
                            "CANDIDATE_DECISION_MISMATCH",
                            f"expected {expected_decision.value}, got "
                            f"{self._candidate_decision(report).value}",
                        )
                    ]
                )
            evidence_pairs = {
                (item.source_id, item.reference) for item in answer.evidence
            }
            if (draft["source_id"], draft["reference"]) not in evidence_pairs:
                issues.append(
                    CandidateIssue(
                        item_id,
                        "CURATED_EVIDENCE_MISSING",
                        "candidate must include the curated source and reference",
                    )
                )
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(CandidateIssue(item_id, "CANDIDATE_ANSWER_INVALID", str(exc)))
        return issues

    def _validate_preference_candidate(
        self,
        item_id: str,
        record: dict[str, Any],
        draft: dict[str, Any],
        pipeline: InferenceReviewPipeline,
    ) -> list[CandidateIssue]:
        issues: list[CandidateIssue] = []
        try:
            chosen = MoralAnswer.from_dict(record["chosen"])
            rejected = MoralAnswer.from_dict(record["rejected"])
            if record.get("prompt") != draft["prompt"]:
                issues.append(
                    CandidateIssue(
                        item_id,
                        "CANDIDATE_PROMPT_MISMATCH",
                        "preference prompt must exactly match the curated prompt",
                    )
                )
            if chosen.request_text != draft["prompt"] or rejected.request_text != draft["prompt"]:
                issues.append(
                    CandidateIssue(
                        item_id,
                        "ANSWER_PROMPT_MISMATCH",
                        "both answers must exactly match the preference prompt",
                    )
                )
            report = pipeline.review(chosen)
            expected_decision = PipelineDecision(
                str(record.get("expected_decision", "release"))
            )
            if self._candidate_decision(report) is not expected_decision:
                issues.extend(
                    CandidateIssue(item_id, f"CHOSEN_{issue.code}", issue.message)
                    for issue in report.issues
                    if issue.code != "CITATION_EVIDENCE_UNREVIEWED"
                )
                if not any(
                    issue.code != "CITATION_EVIDENCE_UNREVIEWED" for issue in report.issues
                ):
                    issues.append(
                        CandidateIssue(
                            item_id,
                            "CHOSEN_DECISION_MISMATCH",
                            f"expected {expected_decision.value}, got release",
                        )
                    )
            if record["chosen"] == record["rejected"]:
                issues.append(
                    CandidateIssue(
                        item_id, "IDENTICAL_PREFERENCE_ANSWERS", "answers must differ"
                    )
                )
            evidence_pairs = {
                (item.source_id, item.reference) for item in chosen.evidence
            }
            if (draft["source_id"], draft["reference"]) not in evidence_pairs:
                issues.append(
                    CandidateIssue(
                        item_id,
                        "CURATED_EVIDENCE_MISSING",
                        "chosen answer must include the curated source and reference",
                    )
                )
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(CandidateIssue(item_id, "PREFERENCE_CANDIDATE_INVALID", str(exc)))
        return issues

    @staticmethod
    def _validate_provenance(
        item_id: str,
        record: dict[str, Any],
        draft: dict[str, Any],
        sources: dict[str, dict[str, Any]],
    ) -> list[CandidateIssue]:
        issues: list[CandidateIssue] = []
        provenance = record.get("provenance")
        if not isinstance(provenance, dict):
            return [CandidateIssue(item_id, "PROVENANCE_MISSING", "provenance is required")]
        source_id = str(draft["source_id"])
        author_id = str(provenance.get("author_id", "")).strip()
        if not author_id:
            issues.append(
                CandidateIssue(item_id, "AUTHOR_ID_MISSING", "provenance.author_id is required")
            )
        try:
            created_at = datetime.fromisoformat(
                str(provenance.get("created_at", "")).replace("Z", "+00:00")
            )
            if created_at.tzinfo is None:
                raise ValueError("timezone required")
        except ValueError:
            issues.append(
                CandidateIssue(
                    item_id,
                    "CREATED_AT_INVALID",
                    "provenance.created_at must be timezone-aware",
                )
            )
        source_ids = provenance.get("source_ids", [])
        if not isinstance(source_ids, list) or source_id not in source_ids:
            issues.append(
                CandidateIssue(
                    item_id,
                    "CURATED_SOURCE_NOT_IN_PROVENANCE",
                    f"provenance must include {source_id}",
                )
            )
        decisions = provenance.get("license_check_ids", {})
        for declared_source in (source_ids if isinstance(source_ids, list) else []):
            source = sources.get(str(declared_source))
            if source is None:
                issues.append(
                    CandidateIssue(
                        item_id,
                        "SOURCE_NOT_APPROVED",
                        f"provenance source is not approved: {declared_source}",
                    )
                )
                continue
            expected = source["approval"]["decision_id"]
            if not isinstance(decisions, dict) or decisions.get(declared_source) != expected:
                issues.append(
                    CandidateIssue(
                        item_id,
                        "LICENSE_DECISION_MISMATCH",
                        f"license_check_ids[{declared_source}] must equal {expected}",
                    )
                )
        return issues

    @staticmethod
    def _category(value: object) -> str:
        return re.sub(r"[\s-]+", "_", str(value).strip().casefold())

    @staticmethod
    def _candidate_decision(report: Any) -> PipelineDecision:
        return strongest_decision(
            issue
            for issue in report.issues
            if issue.code != "CITATION_EVIDENCE_UNREVIEWED"
        )

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            temporary = Path(handle.name)
        temporary.replace(path)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            temporary = Path(handle.name)
        temporary.replace(path)
