"""Review-aware JSONL loading and training-data preparation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .pipeline import InferenceReviewPipeline
from .render import build_sft_messages, render_moral_answer
from .reviewers import (
    SENSITIVE_REVIEW_CATEGORIES,
    is_active_reviewer,
    reviewer_is_qualified,
)
from .schemas import MoralAnswer, PipelineDecision


@dataclass(frozen=True, slots=True)
class DatasetIssue:
    record_id: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class DatasetValidationReport:
    accepted: int
    rejected: int
    issues: tuple[DatasetIssue, ...]

    @property
    def passed(self) -> bool:
        return self.rejected == 0


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL record at line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} must be an object")
            records.append(value)
    return records


class ReviewedDatasetValidator:
    SENSITIVE_CATEGORIES = SENSITIVE_REVIEW_CATEGORIES

    def __init__(
        self,
        pipeline: InferenceReviewPipeline,
        *,
        source_registry: dict[str, Any] | None = None,
        reviewer_registry: dict[str, Any] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.source_registry = source_registry
        self.reviewer_registry = reviewer_registry

    def validate_sft(self, records: list[dict[str, Any]]) -> DatasetValidationReport:
        issues: list[DatasetIssue] = []
        accepted = 0
        seen_ids: set[str] = set()
        for index, record in enumerate(records):
            record_id = str(record.get("record_id", f"row-{index}"))
            row_issues = self._validate_common_review(record, record_id)
            row_issues.extend(self._validate_sft_metadata(record, record_id))
            row_issues.extend(self._validate_governance(record, record_id))
            if record_id in seen_ids:
                row_issues.append(
                    DatasetIssue(record_id, "DUPLICATE_RECORD_ID", "record_id is not unique")
                )
            seen_ids.add(record_id)
            try:
                answer = MoralAnswer.from_dict(record["answer"])
                report = self.pipeline.review(answer)
                if report.decision is not PipelineDecision.RELEASE:
                    row_issues.extend(
                        DatasetIssue(record_id, issue.code, issue.message)
                        for issue in report.issues
                    )
            except (KeyError, TypeError, ValueError) as exc:
                row_issues.append(DatasetIssue(record_id, "INVALID_MORAL_ANSWER", str(exc)))
            if row_issues:
                issues.extend(row_issues)
            else:
                accepted += 1
        return DatasetValidationReport(accepted, len(records) - accepted, tuple(issues))

    def validate_preferences(self, records: list[dict[str, Any]]) -> DatasetValidationReport:
        issues: list[DatasetIssue] = []
        accepted = 0
        seen_ids: set[str] = set()
        for index, record in enumerate(records):
            record_id = str(record.get("pair_id", f"row-{index}"))
            row_issues: list[DatasetIssue] = []
            row_issues.extend(self._validate_preference_metadata(record, record_id))
            row_issues.extend(self._validate_governance(record, record_id))
            if record.get("status") != "accepted":
                row_issues.append(
                    DatasetIssue(record_id, "STATUS_NOT_ACCEPTED", "pair is not accepted")
                )
            if not re.fullmatch(r"DPO-[A-Z0-9_-]+", record_id):
                row_issues.append(
                    DatasetIssue(
                        record_id,
                        "INVALID_PAIR_ID",
                        "pair_id must match DPO-[A-Z0-9_-]+",
                    )
                )
            if record_id in seen_ids:
                row_issues.append(
                    DatasetIssue(record_id, "DUPLICATE_PAIR_ID", "pair_id is not unique")
                )
            seen_ids.add(record_id)

            reviews = record.get("reviews", [])
            reviewer_ids = {item.get("reviewer_id") for item in reviews if isinstance(item, dict)}
            if len(reviews) < 2 or len(reviewer_ids) < 2:
                row_issues.append(
                    DatasetIssue(
                        record_id,
                        "PAIR_DUAL_REVIEW_MISSING",
                        "preference pairs require two independent reviewers",
                    )
                )
            if any(
                not item.get("independent", False) for item in reviews if isinstance(item, dict)
            ):
                row_issues.append(
                    DatasetIssue(
                        record_id,
                        "PAIR_REVIEW_NOT_INDEPENDENT",
                        "all preference reviews must be independent",
                    )
                )
            for review in reviews:
                if not isinstance(review, dict):
                    continue
                if review.get("decision") != "approve_chosen":
                    row_issues.append(
                        DatasetIssue(
                            record_id,
                            "PAIR_REVIEW_NOT_APPROVED",
                            "countable preference reviews must approve the chosen answer",
                        )
                    )
                if not review.get("affiliations_disclosed", False):
                    row_issues.append(
                        DatasetIssue(
                            record_id,
                            "PAIR_AFFILIATION_NOT_DISCLOSED",
                            "preference reviewer affiliation disclosure is required",
                        )
                    )
                if not str(review.get("rationale", "")).strip():
                    row_issues.append(
                        DatasetIssue(
                            record_id,
                            "PAIR_REVIEW_RATIONALE_MISSING",
                            "every preference review requires a rationale",
                        )
                    )

            reasons = " ".join(record.get("preference_reasons", [])).casefold()
            if any(
                phrase in reasons
                for phrase in (
                    "agrees with my denomination",
                    "agrees with the sda",
                    "agrees with adventist",
                    "matches the reviewer's affiliation",
                    "matches the reviewer affiliation",
                )
            ):
                row_issues.append(
                    DatasetIssue(
                        record_id,
                        "AFFILIATION_PREFERENCE",
                        "denominational or reviewer affiliation cannot justify a preference",
                    )
                )

            try:
                chosen = MoralAnswer.from_dict(record["chosen"])
                rejected = MoralAnswer.from_dict(record["rejected"])
                prompt = str(record.get("prompt", ""))
                if chosen.request_text != prompt or rejected.request_text != prompt:
                    row_issues.append(
                        DatasetIssue(
                            record_id,
                            "PAIR_PROMPT_MISMATCH",
                            "prompt must exactly match chosen and rejected request_text",
                        )
                    )
                if record["chosen"] == record["rejected"]:
                    row_issues.append(
                        DatasetIssue(
                            record_id,
                            "PAIR_IDENTICAL_ANSWERS",
                            "chosen and rejected answers must differ",
                        )
                    )
                chosen_report = self.pipeline.review(chosen)
                expected_decision = PipelineDecision(
                    str(record.get("expected_decision", "release"))
                )
                if chosen_report.decision is not expected_decision:
                    row_issues.extend(
                        DatasetIssue(record_id, f"CHOSEN_{issue.code}", issue.message)
                        for issue in chosen_report.issues
                    )
                    if not chosen_report.issues:
                        row_issues.append(
                            DatasetIssue(
                                record_id,
                                "CHOSEN_DECISION_MISMATCH",
                                f"expected {expected_decision.value}, got release",
                            )
                        )
            except (KeyError, TypeError, ValueError) as exc:
                row_issues.append(DatasetIssue(record_id, "INVALID_PREFERENCE_ANSWER", str(exc)))

            if row_issues:
                issues.extend(row_issues)
            else:
                accepted += 1
        return DatasetValidationReport(accepted, len(records) - accepted, tuple(issues))

    def validate_evals(self, records: list[dict[str, Any]]) -> DatasetValidationReport:
        issues: list[DatasetIssue] = []
        accepted = 0
        seen_ids: set[str] = set()
        for index, record in enumerate(records):
            record_id = str(record.get("case_id", f"row-{index}"))
            row_issues = self._validate_common_review(record, record_id)
            row_issues.extend(self._validate_sft_metadata(record, record_id, id_prefix="EVAL"))
            row_issues.extend(self._validate_governance(record, record_id))
            if record_id in seen_ids:
                row_issues.append(
                    DatasetIssue(record_id, "DUPLICATE_CASE_ID", "case_id is not unique")
                )
            seen_ids.add(record_id)
            try:
                answer = MoralAnswer.from_dict(record["answer"])
                expected_decision = PipelineDecision(
                    str(record.get("expected_decision", "release"))
                )
                actual_decision = self.pipeline.review(answer).decision
                if actual_decision is not expected_decision:
                    row_issues.append(
                        DatasetIssue(
                            record_id,
                            "EVAL_DECISION_MISMATCH",
                            f"expected {expected_decision.value}, got {actual_decision.value}",
                        )
                    )
            except (KeyError, TypeError, ValueError) as exc:
                row_issues.append(DatasetIssue(record_id, "INVALID_EVAL_ANSWER", str(exc)))
            if row_issues:
                issues.extend(row_issues)
            else:
                accepted += 1
        return DatasetValidationReport(accepted, len(records) - accepted, tuple(issues))

    @staticmethod
    def _validate_sft_metadata(
        record: dict[str, Any], record_id: str, *, id_prefix: str = "SFT"
    ) -> list[DatasetIssue]:
        issues: list[DatasetIssue] = []
        if not re.fullmatch(rf"{id_prefix}-[A-Z0-9_-]+", record_id):
            issues.append(
                DatasetIssue(
                    record_id,
                    "INVALID_RECORD_ID",
                    f"record ID must match {id_prefix}-[A-Z0-9_-]+",
                )
            )
        if not str(record.get("scenario_id", "")).strip():
            issues.append(DatasetIssue(record_id, "SCENARIO_ID_MISSING", "scenario_id is required"))
        provenance = record.get("provenance")
        if not isinstance(provenance, dict):
            issues.append(
                DatasetIssue(record_id, "PROVENANCE_MISSING", "provenance object is required")
            )
            return issues
        for field in ("author_id", "created_at"):
            if not str(provenance.get(field, "")).strip():
                issues.append(
                    DatasetIssue(
                        record_id,
                        "PROVENANCE_FIELD_MISSING",
                        f"provenance.{field} is required",
                    )
                )
        if not provenance.get("license_check_ids") and not provenance.get("license_check_id"):
            issues.append(
                DatasetIssue(
                    record_id,
                    "PROVENANCE_FIELD_MISSING",
                    "provenance.license_check_ids is required",
                )
            )
        try:
            created_at = datetime.fromisoformat(
                str(provenance.get("created_at", "")).replace("Z", "+00:00")
            )
            if created_at.tzinfo is None:
                raise ValueError("timezone required")
        except ValueError:
            issues.append(
                DatasetIssue(
                    record_id,
                    "PROVENANCE_TIMESTAMP_INVALID",
                    "provenance.created_at must be a timezone-aware ISO timestamp",
                )
            )
        source_ids = provenance.get("source_ids", [])
        if not isinstance(source_ids, list) or not source_ids:
            issues.append(
                DatasetIssue(
                    record_id,
                    "PROVENANCE_SOURCES_MISSING",
                    "provenance.source_ids must be a non-empty list",
                )
            )
        else:
            answer = record.get("answer", {})
            evidence = answer.get("evidence", []) if isinstance(answer, dict) else []
            answer_sources = {
                str(item.get("source_id")) for item in evidence if isinstance(item, dict)
            }
            if not answer_sources <= set(source_ids):
                issues.append(
                    DatasetIssue(
                        record_id,
                        "PROVENANCE_SOURCE_MISMATCH",
                        "every answer evidence source must be in provenance.source_ids",
                    )
                )
        return issues

    def _validate_governance(
        self, record: dict[str, Any], record_id: str
    ) -> list[DatasetIssue]:
        issues: list[DatasetIssue] = []
        category = re.sub(
            r"[\s-]+", "_", str(record.get("category", "")).strip().casefold()
        )
        if category in self.SENSITIVE_CATEGORIES and record.get("high_impact") is not True:
            issues.append(
                DatasetIssue(
                    record_id,
                    "SENSITIVE_CATEGORY_NOT_HIGH_IMPACT",
                    f"{category} records must be marked high_impact",
                )
            )
        if self.source_registry is not None:
            approved = {
                str(item.get("source_id")): item
                for item in self.source_registry.get("sources", [])
                if item.get("status") == "approved"
            }
            provenance = record.get("provenance", {})
            source_ids = provenance.get("source_ids", []) if isinstance(provenance, dict) else []
            decisions = (
                provenance.get("license_check_ids", {}) if isinstance(provenance, dict) else {}
            )
            for source_id in source_ids:
                source = approved.get(str(source_id))
                if source is None:
                    issues.append(
                        DatasetIssue(
                            record_id,
                            "SOURCE_NOT_APPROVED",
                            f"provenance source is not approved: {source_id}",
                        )
                    )
                    continue
                expected = source.get("approval", {}).get("decision_id")
                if decisions.get(source_id) != expected:
                    issues.append(
                        DatasetIssue(
                            record_id,
                            "LICENSE_DECISION_MISMATCH",
                            f"license_check_ids[{source_id}] must equal {expected}",
                        )
                    )
        if self.reviewer_registry is not None:
            allowed = {
                str(item)
                for item in self.reviewer_registry.get("qualified_categories", [])
            }
            reviewers = {
                str(item.get("reviewer_id")): item
                for item in self.reviewer_registry.get("reviewers", [])
            }
            author_id = str(record.get("provenance", {}).get("author_id", ""))
            for review in record.get("reviews", []):
                if not isinstance(review, dict):
                    continue
                reviewer_id = str(review.get("reviewer_id", ""))
                reviewer = reviewers.get(reviewer_id)
                if (
                    reviewer is None
                    or not is_active_reviewer(
                        reviewer, allowed_qualifications=allowed
                    )
                ):
                    issues.append(
                        DatasetIssue(
                            record_id,
                            "REVIEWER_NOT_REGISTERED",
                            f"reviewer is not active, disclosed, and attested: {reviewer_id}",
                        )
                    )
                elif not reviewer_is_qualified(reviewer, record.get("category", "")):
                    issues.append(
                        DatasetIssue(
                            record_id,
                            "REVIEWER_NOT_QUALIFIED",
                            f"reviewer is not qualified for {record.get('category', '')}: {reviewer_id}",
                        )
                    )
                if reviewer_id == author_id:
                    issues.append(
                        DatasetIssue(
                            record_id,
                            "REVIEWER_IS_AUTHOR",
                            "an author cannot count as an independent reviewer",
                        )
                    )
        return issues

    @staticmethod
    def _validate_preference_metadata(record: dict[str, Any], record_id: str) -> list[DatasetIssue]:
        issues: list[DatasetIssue] = []
        if not str(record.get("scenario_id", "")).strip():
            issues.append(DatasetIssue(record_id, "SCENARIO_ID_MISSING", "scenario_id is required"))
        if not isinstance(record.get("high_impact"), bool):
            issues.append(
                DatasetIssue(
                    record_id,
                    "HIGH_IMPACT_FLAG_MISSING",
                    "high_impact must be an explicit boolean",
                )
            )
        provenance = record.get("provenance")
        if not isinstance(provenance, dict):
            issues.append(
                DatasetIssue(record_id, "PROVENANCE_MISSING", "provenance object is required")
            )
            return issues
        for field in ("author_id", "created_at"):
            if not str(provenance.get(field, "")).strip():
                issues.append(
                    DatasetIssue(
                        record_id,
                        "PROVENANCE_FIELD_MISSING",
                        f"provenance.{field} is required",
                    )
                )
        if not provenance.get("license_check_ids") and not provenance.get("license_check_id"):
            issues.append(
                DatasetIssue(
                    record_id,
                    "PROVENANCE_FIELD_MISSING",
                    "provenance.license_check_ids is required",
                )
            )
        try:
            created_at = datetime.fromisoformat(
                str(provenance.get("created_at", "")).replace("Z", "+00:00")
            )
            if created_at.tzinfo is None:
                raise ValueError("timezone required")
        except ValueError:
            issues.append(
                DatasetIssue(
                    record_id,
                    "PROVENANCE_TIMESTAMP_INVALID",
                    "provenance.created_at must be a timezone-aware ISO timestamp",
                )
            )
        source_ids = provenance.get("source_ids", [])
        if not isinstance(source_ids, list) or not source_ids:
            issues.append(
                DatasetIssue(
                    record_id,
                    "PROVENANCE_SOURCES_MISSING",
                    "provenance.source_ids must be a non-empty list",
                )
            )
        return issues

    @staticmethod
    def _validate_common_review(record: dict[str, Any], record_id: str) -> list[DatasetIssue]:
        issues: list[DatasetIssue] = []
        if record.get("status") != "accepted":
            issues.append(DatasetIssue(record_id, "STATUS_NOT_ACCEPTED", "record is not accepted"))
        reviews = record.get("reviews", [])
        required = 2 if record.get("high_impact") else 1
        reviewer_ids = {item.get("reviewer_id") for item in reviews if isinstance(item, dict)}
        if len(reviews) < required or len(reviewer_ids) < required:
            issues.append(
                DatasetIssue(
                    record_id,
                    "REVIEW_COVERAGE_MISSING",
                    f"record requires {required} independent reviewer(s)",
                )
            )
        for review in reviews:
            if not isinstance(review, dict) or not review.get("independent", False):
                issues.append(
                    DatasetIssue(record_id, "REVIEW_NOT_INDEPENDENT", "review must be independent")
                )
            if not isinstance(review, dict) or not str(review.get("reviewer_id", "")).strip():
                issues.append(
                    DatasetIssue(
                        record_id,
                        "REVIEWER_ID_MISSING",
                        "every countable review requires a reviewer_id",
                    )
                )
            if not review.get("affiliations_disclosed", False):
                issues.append(
                    DatasetIssue(
                        record_id,
                        "AFFILIATION_NOT_DISCLOSED",
                        "reviewer affiliation disclosure is required",
                    )
                )
            if review.get("decision") != "approve":
                issues.append(
                    DatasetIssue(
                        record_id, "REVIEW_NOT_APPROVED", "countable records require approval"
                    )
                )
            if not str(review.get("rationale", "")).strip():
                issues.append(
                    DatasetIssue(
                        record_id,
                        "REVIEW_RATIONALE_MISSING",
                        "every review requires a rationale",
                    )
                )
        return issues


def materialize_sft_record(record: dict[str, Any], *, system_prompt: str) -> dict[str, Any]:
    answer = MoralAnswer.from_dict(record["answer"])
    return {
        "record_id": record["record_id"],
        "scenario_id": record["scenario_id"],
        "messages": build_sft_messages(answer, system_prompt=system_prompt),
    }


def materialize_preference_record(record: dict[str, Any]) -> dict[str, str]:
    chosen = MoralAnswer.from_dict(record["chosen"])
    rejected = MoralAnswer.from_dict(record["rejected"])
    return {
        "pair_id": record["pair_id"],
        "prompt": record["prompt"],
        "chosen": render_moral_answer(chosen),
        "rejected": render_moral_answer(rejected),
    }
