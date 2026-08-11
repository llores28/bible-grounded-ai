"""Reviewer qualification, recruitment readiness, and blinded handoff kits."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .evidence_store import file_sha256
from .registry import load_json

SENSITIVE_REVIEW_CATEGORIES = frozenset(
    {"prophecy", "abuse", "violence", "force", "disputed_doctrine"}
)
_REVIEWER_ID = re.compile(r"REVIEWER-[A-Z0-9_-]+")


def normalize_review_category(value: object) -> str:
    return re.sub(r"[\s-]+", "_", str(value).strip().casefold())


def reviewer_record_issues(
    reviewer: dict[str, Any], *, allowed_qualifications: set[str]
) -> tuple[str, ...]:
    """Return fail-closed registry defects for one reviewer record."""
    issues: list[str] = []
    reviewer_id = str(reviewer.get("reviewer_id", ""))
    if not _REVIEWER_ID.fullmatch(reviewer_id):
        issues.append("reviewer_id must match REVIEWER-[A-Z0-9_-]+")
    if reviewer.get("status") not in {"active", "inactive"}:
        issues.append("status must be active or inactive")
    affiliations = reviewer.get("affiliations")
    if not isinstance(affiliations, list) or not affiliations or not all(
        isinstance(item, str) and item.strip() for item in affiliations
    ):
        issues.append("affiliations must be a non-empty list of disclosures or 'none'")
    qualifications = reviewer.get("qualified_categories")
    if not isinstance(qualifications, list) or not qualifications:
        issues.append("qualified_categories must be a non-empty list")
    elif len(set(map(str, qualifications))) != len(qualifications):
        issues.append("qualified_categories must be unique")
    elif any(str(item) not in allowed_qualifications for item in qualifications):
        issues.append("qualified_categories contains an unregistered qualification")
    attested = reviewer.get("independence_attested_on")
    try:
        attested_on = date.fromisoformat(str(attested))
        if attested_on > date.today():
            issues.append("independence_attested_on cannot be in the future")
    except ValueError:
        issues.append("independence_attested_on must be an ISO date")
    if reviewer.get("status") == "active" and reviewer.get(
        "affiliations_disclosed"
    ) is not True:
        issues.append("active reviewers must affirm affiliations_disclosed")
    return tuple(issues)


def is_active_reviewer(
    reviewer: dict[str, Any], *, allowed_qualifications: set[str]
) -> bool:
    return reviewer.get("status") == "active" and not reviewer_record_issues(
        reviewer, allowed_qualifications=allowed_qualifications
    )


def reviewer_is_qualified(reviewer: dict[str, Any], category: object) -> bool:
    """General never substitutes for a named sensitive qualification."""
    normalized = normalize_review_category(category)
    qualifications = {
        normalize_review_category(item)
        for item in reviewer.get("qualified_categories", [])
    }
    if normalized in SENSITIVE_REVIEW_CATEGORIES:
        return normalized in qualifications
    return "general" in qualifications or normalized in qualifications


@dataclass(frozen=True, slots=True)
class ReviewerReadinessIssue:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ReviewerReadinessReport:
    passed: bool
    registered_count: int
    active_valid_count: int
    required_capacity: dict[str, int]
    available_capacity: dict[str, int]
    scenario_counts: dict[str, int]
    adjudication_reserve: dict[str, int]
    issues: tuple[ReviewerReadinessIssue, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReviewerWorkflow:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def audit_readiness(self) -> ReviewerReadinessReport:
        registry = load_json(self.root / "configs/reviewers.json")
        queue = load_json(self.root / "configs/pilot/draft_scenarios.json")
        reviewers = registry.get("reviewers", [])
        allowed = {str(item) for item in registry.get("qualified_categories", [])}
        issues: list[ReviewerReadinessIssue] = []
        if not isinstance(reviewers, list):
            reviewers = []
            issues.append(
                ReviewerReadinessIssue(
                    "REVIEWER_REGISTRY_INVALID", "reviewers must be a list"
                )
            )
        ids: list[str] = []
        for index, reviewer in enumerate(reviewers):
            if not isinstance(reviewer, dict):
                issues.append(
                    ReviewerReadinessIssue(
                        "REVIEWER_RECORD_INVALID", f"reviewers[{index}] must be an object"
                    )
                )
                continue
            reviewer_id = str(reviewer.get("reviewer_id", f"row-{index}"))
            ids.append(reviewer_id)
            for detail in reviewer_record_issues(
                reviewer, allowed_qualifications=allowed
            ):
                issues.append(
                    ReviewerReadinessIssue(
                        "REVIEWER_RECORD_INVALID", f"{reviewer_id}: {detail}"
                    )
                )
        if len(set(ids)) != len(ids):
            issues.append(
                ReviewerReadinessIssue(
                    "DUPLICATE_REVIEWER_ID", "reviewer IDs must be unique"
                )
            )
        active = [
            reviewer
            for reviewer in reviewers
            if isinstance(reviewer, dict)
            and is_active_reviewer(reviewer, allowed_qualifications=allowed)
        ]
        if len(active) < 2:
            issues.append(
                ReviewerReadinessIssue(
                    "ACTIVE_REVIEWERS_MISSING",
                    f"active valid reviewers: {len(active)}/2",
                )
            )

        required_capacity: dict[str, int] = {}
        scenario_counts: dict[str, int] = {}
        for split in ("sft", "preferences", "evals"):
            for item in queue.get(split, []):
                category = normalize_review_category(item.get("category", ""))
                lane = category if category in SENSITIVE_REVIEW_CATEGORIES else "general"
                required = 2 if split == "preferences" or lane != "general" else 1
                required_capacity[lane] = max(required_capacity.get(lane, 0), required)
                scenario_counts[lane] = scenario_counts.get(lane, 0) + 1

        available_capacity = {
            lane: sum(reviewer_is_qualified(reviewer, lane) for reviewer in active)
            for lane in required_capacity
        }
        for lane, required in sorted(required_capacity.items()):
            available = available_capacity[lane]
            if available < required:
                issues.append(
                    ReviewerReadinessIssue(
                        "QUALIFIED_REVIEWER_COVERAGE_MISSING",
                        f"{lane}: {available}/{required} qualified active reviewers",
                    )
                )
        if active and registry.get("status") != "active":
            issues.append(
                ReviewerReadinessIssue(
                    "REVIEWER_REGISTRY_STATUS_INVALID",
                    "registry status must be active when active reviewers are used",
                )
            )
        adjudication_reserve = {
            lane: max(0, available_capacity[lane] - required)
            for lane, required in required_capacity.items()
        }
        warnings = tuple(
            f"{lane} has no third qualified reviewer reserved for disagreement adjudication"
            for lane, reserve in sorted(adjudication_reserve.items())
            if reserve < 1
        )
        return ReviewerReadinessReport(
            passed=not issues,
            registered_count=len(reviewers),
            active_valid_count=len(active),
            required_capacity=dict(sorted(required_capacity.items())),
            available_capacity=dict(sorted(available_capacity.items())),
            scenario_counts=dict(sorted(scenario_counts.items())),
            adjudication_reserve=dict(sorted(adjudication_reserve.items())),
            issues=tuple(issues),
            warnings=warnings,
        )

    def build_recruitment_kit(
        self, output_dir: str | Path = "data/reviewer_kits/recruitment"
    ) -> dict[str, Any]:
        report = self.audit_readiness()
        destination = self.root / output_dir
        destination.mkdir(parents=True, exist_ok=True)
        registry_path = self.root / "configs/reviewers.json"
        queue_path = self.root / "configs/pilot/draft_scenarios.json"
        requirements = {
            "schema_version": "1.0",
            "purpose": "Recruit real independent pilot reviewers; this file is not an approval.",
            "reviewer_registry_sha256": file_sha256(registry_path),
            "draft_queue_sha256": file_sha256(queue_path),
            "readiness": report.to_dict(),
        }
        registration = {
            "reviewer_id": "REVIEWER-REPLACE_ME",
            "status": "inactive",
            "affiliations_disclosed": False,
            "affiliations": ["REPLACE_WITH_DISCLOSURE_OR_NONE"],
            "independence_attested_on": "REPLACE_WITH_YYYY-MM-DD",
            "qualified_categories": ["general"],
        }
        readme = """# Independent pilot reviewer recruitment kit

This kit recruits real people; it does not create approvals. Do not invent an identity, reuse another person's attestation, or activate a reviewer without their informed participation.

1. Read `docs/PILOT_REVIEW_WORKFLOW.md` in the repository.
2. Copy `reviewer-registration.template.json` and replace every placeholder.
3. Disclose denominational, institutional, project, and relevant personal affiliations; use the literal string `none` only when accurate.
4. Claim only categories you are competent to review. `general` does not qualify a reviewer for prophecy, abuse, violence, force, or disputed doctrine.
5. Attest to independent blinded first-pass review. A candidate author cannot review that candidate.
6. Add the reviewed registration to `configs/reviewers.json`, set the registry status to `active`, run `audit-reviewers`, then create assignments and per-reviewer kits.

Two qualified independent reviewers are mandatory for every preference pair and every sensitive case. A third qualified person is recommended in each lane so disagreements can be adjudicated without reusing an assigned reviewer.
"""
        self._write_json(destination / "requirements.json", requirements)
        self._write_json(
            destination / "reviewer-registration.template.json", registration
        )
        self._write_text(destination / "README.md", readme)
        return {
            "status": "reviewer_recruitment_kit_built",
            "output_dir": str(destination),
            "readiness_passed": report.passed,
            "files": {
                path.name: file_sha256(path)
                for path in sorted(destination.iterdir())
                if path.is_file()
            },
        }

    def export_assigned_kits(
        self,
        *,
        packets_path: str | Path = "data/pilot/candidate_review_packets.jsonl",
        assignments_path: str | Path = "data/pilot/reviewer_assignments.json",
        output_dir: str | Path = "data/reviewer_kits/assigned",
    ) -> dict[str, Any]:
        readiness = self.audit_readiness()
        if not readiness.passed:
            raise ValueError("reviewer readiness must pass before exporting assigned kits")
        packet_file = self.root / packets_path
        assignment_file = self.root / assignments_path
        if not packet_file.is_file() or not assignment_file.is_file():
            raise ValueError("build review packets and assignments before exporting kits")
        packets = self._read_jsonl(packet_file)
        assignments = load_json(assignment_file)
        if assignments.get("candidate_packets_sha256") != file_sha256(packet_file):
            raise ValueError("assignments do not bind to the current packet file")
        registry_path = self.root / "configs/reviewers.json"
        if assignments.get("reviewer_registry_sha256") != file_sha256(registry_path):
            raise ValueError("assignments do not bind to the current reviewer registry")
        registry = load_json(registry_path)
        reviewers = {
            str(item["reviewer_id"]): item for item in registry.get("reviewers", [])
        }
        packet_by_id = {str(item["item_id"]): item for item in packets}
        if len(packet_by_id) != len(packets):
            raise ValueError("candidate review packet item IDs must be unique")
        for packet in packets:
            claimed = str(packet.get("packet_sha256", ""))
            unsigned = {key: value for key, value in packet.items() if key != "packet_sha256"}
            canonical = json.dumps(
                unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != claimed:
                raise ValueError(f"candidate review packet digest mismatch: {packet.get('item_id')}")
        by_reviewer: dict[str, list[dict[str, Any]]] = {}
        for assignment in assignments.get("assignments", []):
            item_id = str(assignment.get("item_id", ""))
            packet = packet_by_id.get(item_id)
            if packet is None or assignment.get("packet_sha256") != packet.get(
                "packet_sha256"
            ):
                raise ValueError(f"assignment is stale or invalid: {item_id}")
            reviewer_ids = assignment.get("reviewer_ids", [])
            if not isinstance(reviewer_ids, list) or len(set(reviewer_ids)) != len(
                reviewer_ids
            ):
                raise ValueError(f"assignment reviewer IDs are invalid: {item_id}")
            required = int(packet.get("required_independent_reviewers", 0))
            if len(reviewer_ids) != required:
                raise ValueError(
                    f"assignment reviewer coverage is invalid: {item_id} "
                    f"({len(reviewer_ids)}/{required})"
                )
            for reviewer_id in reviewer_ids:
                reviewer = reviewers.get(str(reviewer_id))
                if reviewer is None or not reviewer_is_qualified(
                    reviewer, packet.get("category", "")
                ):
                    raise ValueError(
                        f"reviewer is not qualified for assigned category: {reviewer_id}/{item_id}"
                    )
                by_reviewer.setdefault(str(reviewer_id), []).append(packet)
        assignment_digest = file_sha256(assignment_file)
        version_dir = self.root / output_dir / assignment_digest[:12]
        version_dir.mkdir(parents=True, exist_ok=True)
        bundles: dict[str, dict[str, Any]] = {}
        for reviewer_id, reviewer_packets in sorted(by_reviewer.items()):
            reviewer_dir = version_dir / reviewer_id
            reviewer_dir.mkdir(parents=True, exist_ok=True)
            packet_path = reviewer_dir / "packets.jsonl"
            template_path = reviewer_dir / "reviews.template.jsonl"
            manifest_path = reviewer_dir / "manifest.json"
            readme_path = reviewer_dir / "README.md"
            self._write_jsonl(packet_path, reviewer_packets)
            templates = [
                {
                    "review_id": f"REVIEW-{reviewer_id.removeprefix('REVIEWER-')}-{packet['item_id']}",
                    "item_id": packet["item_id"],
                    "reviewer_id": reviewer_id,
                    "packet_sha256": packet["packet_sha256"],
                    "decision": "REPLACE_WITH_APPROVE_REVISE_OR_REJECT",
                    "rationale": "REPLACE_WITH_INDEPENDENT_RATIONALE",
                    "required_corrections": [],
                    "reviewed_at": "REPLACE_WITH_TIMEZONE_AWARE_ISO_TIMESTAMP",
                    "independent_blind_attestation": False,
                    "affiliations_disclosed": False,
                }
                for packet in reviewer_packets
            ]
            self._write_jsonl(template_path, templates)
            self._write_text(
                readme_path,
                "# Blinded pilot review kit\n\n"
                f"Assigned reviewer: `{reviewer_id}`\n\n"
                "Review each packet without seeing another reviewer's decision. Verify the exact quotation, bounded application, uncertainty, counter-readings, commandment assessments, affected people, safety referrals, and expected release/refuse/escalate behavior. Replace every placeholder and explicitly change both attestations to `true` only when accurate. Return only the completed JSONL; do not edit packet hashes.\n",
            )
            manifest = {
                "schema_version": "1.0",
                "reviewer_id": reviewer_id,
                "assignment_file_sha256": assignment_digest,
                "reviewer_registry_sha256": file_sha256(registry_path),
                "packet_count": len(reviewer_packets),
                "packet_file_sha256": file_sha256(packet_path),
                "template_file_sha256": file_sha256(template_path),
            }
            self._write_json(manifest_path, manifest)
            zip_path = version_dir / f"{reviewer_id}.zip"
            self._write_deterministic_zip(
                zip_path, (readme_path, manifest_path, packet_path, template_path)
            )
            bundles[reviewer_id] = {
                "packet_count": len(reviewer_packets),
                "zip_path": str(zip_path),
                "zip_sha256": file_sha256(zip_path),
            }
        self._write_json(
            version_dir / "bundle_manifest.json",
            {
                "schema_version": "1.0",
                "assignment_file_sha256": assignment_digest,
                "bundles": bundles,
            },
        )
        return {
            "status": "assigned_reviewer_kits_exported",
            "output_dir": str(version_dir),
            "bundle_count": len(bundles),
            "bundles": bundles,
        }

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(f"blank JSONL record at line {line_number}")
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"JSONL line {line_number} must be an object")
                records.append(record)
        return records

    @staticmethod
    def _write_text(path: Path, value: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False, newline="\n"
        ) as handle:
            handle.write(value)
            temporary = Path(handle.name)
        temporary.replace(path)

    @classmethod
    def _write_json(cls, path: Path, value: Any) -> None:
        cls._write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")

    @classmethod
    def _write_jsonl(cls, path: Path, records: list[dict[str, Any]]) -> None:
        cls._write_text(
            path,
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in records
            ),
        )

    @staticmethod
    def _write_deterministic_zip(zip_path: Path, files: tuple[Path, ...]) -> None:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=zip_path.parent, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(files, key=lambda item: item.name):
                    info = zipfile.ZipInfo(path.name, date_time=(2026, 8, 11, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o100644 << 16
                    archive.writestr(info, path.read_bytes())
            temporary.replace(zip_path)
        finally:
            temporary.unlink(missing_ok=True)
