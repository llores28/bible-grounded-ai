"""Curated pilot draft queue auditing and evidence-backed review packet generation."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .dataset import ReviewedDatasetValidator
from .evidence_store import EvidenceStore, file_sha256
from .registry import load_json


@dataclass(frozen=True, slots=True)
class DraftIssue:
    item_id: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class DraftAuditReport:
    passed: bool
    counts: dict[str, int]
    sensitive_counts: dict[str, int]
    issues: tuple[DraftIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SPLIT_RULES = {
    "sft": ("SFT-DRAFT-", 50),
    "preferences": ("DPO-DRAFT-", 20),
    "evals": ("EVAL-DRAFT-", 25),
}


class PilotDraftWorkflow:
    def __init__(
        self,
        root: str | Path,
        *,
        queue_path: str | Path = "configs/pilot/draft_scenarios.json",
        database_path: str | Path = "data/index/biblical_evidence.sqlite3",
    ) -> None:
        self.root = Path(root).resolve()
        self.queue_path = self.root / queue_path
        self.database_path = self.root / database_path

    def audit(self) -> DraftAuditReport:
        queue = load_json(self.queue_path)
        registry = load_json(self.root / "configs/data/source_registry.json")
        approved = {
            str(item["source_id"]): item
            for item in registry.get("sources", [])
            if item.get("status") == "approved"
        }
        issues: list[DraftIssue] = []
        counts: dict[str, int] = {}
        sensitive_counts = {
            category: 0 for category in sorted(ReviewedDatasetValidator.SENSITIVE_CATEGORIES)
        }
        seen_ids: set[str] = set()
        seen_prompts: set[str] = set()
        if queue.get("status") != "draft_only":
            issues.append(
                DraftIssue("QUEUE", "QUEUE_STATUS_INVALID", "queue status must be draft_only")
            )
        if not self.database_path.is_file():
            issues.append(
                DraftIssue(
                    "QUEUE",
                    "EVIDENCE_DATABASE_MISSING",
                    "run build-evidence before auditing draft scenarios",
                )
            )
            return DraftAuditReport(False, counts, sensitive_counts, tuple(issues))

        with EvidenceStore(self.database_path) as store:
            for split, (prefix, target) in _SPLIT_RULES.items():
                items = queue.get(split, [])
                counts[split] = len(items) if isinstance(items, list) else 0
                if not isinstance(items, list) or len(items) != target:
                    issues.append(
                        DraftIssue(
                            split,
                            "DRAFT_COUNT_MISMATCH",
                            f"{split} must contain exactly {target} drafts",
                        )
                    )
                    continue
                for item in items:
                    item_id = str(item.get("item_id", ""))
                    if not item_id.startswith(prefix):
                        issues.append(
                            DraftIssue(
                                item_id or split,
                                "DRAFT_ID_INVALID",
                                f"item_id must start with {prefix}",
                            )
                        )
                    if item_id in seen_ids:
                        issues.append(
                            DraftIssue(item_id, "DUPLICATE_DRAFT_ID", "item_id is duplicated")
                        )
                    seen_ids.add(item_id)
                    prompt = re.sub(r"\s+", " ", str(item.get("prompt", ""))).strip()
                    prompt_key = prompt.casefold()
                    if not prompt:
                        issues.append(
                            DraftIssue(item_id, "DRAFT_PROMPT_MISSING", "prompt is required")
                        )
                    elif prompt_key in seen_prompts:
                        issues.append(
                            DraftIssue(
                                item_id,
                                "DUPLICATE_DRAFT_PROMPT",
                                "prompt duplicates another pilot draft",
                            )
                        )
                    seen_prompts.add(prompt_key)
                    category = self._category(item.get("category", ""))
                    if category in sensitive_counts:
                        sensitive_counts[category] += 1
                        if item.get("high_impact") is not True:
                            issues.append(
                                DraftIssue(
                                    item_id,
                                    "SENSITIVE_DRAFT_NOT_HIGH_IMPACT",
                                    f"{category} drafts must be high impact",
                                )
                            )
                    source_id = str(item.get("source_id", ""))
                    reference = str(item.get("reference", ""))
                    if source_id not in approved:
                        issues.append(
                            DraftIssue(
                                item_id,
                                "DRAFT_SOURCE_NOT_APPROVED",
                                f"source is not approved: {source_id}",
                            )
                        )
                    if store.get_passage(source_id, reference) is None:
                        issues.append(
                            DraftIssue(
                                item_id,
                                "DRAFT_REFERENCE_NOT_FOUND",
                                f"evidence passage does not exist: {source_id} {reference}",
                            )
                        )
                    if not str(item.get("review_focus", "")).strip():
                        issues.append(
                            DraftIssue(
                                item_id,
                                "REVIEW_FOCUS_MISSING",
                                "review_focus is required",
                            )
                        )
        for category, count in sensitive_counts.items():
            if count == 0:
                issues.append(
                    DraftIssue(
                        "QUEUE",
                        "SENSITIVE_CATEGORY_MISSING",
                        f"no pilot draft covers {category}",
                    )
                )
        return DraftAuditReport(not issues, counts, sensitive_counts, tuple(issues))

    def build_authoring_packets(
        self, output_path: str | Path = "data/pilot/authoring_packets.jsonl"
    ) -> dict[str, Any]:
        report = self.audit()
        if not report.passed:
            raise ValueError(
                "draft audit failed: "
                + " | ".join(f"{issue.code}: {issue.message}" for issue in report.issues[:10])
            )
        queue = load_json(self.queue_path)
        registry = load_json(self.root / "configs/data/source_registry.json")
        sources = {str(item["source_id"]): item for item in registry["sources"]}
        packets: list[dict[str, Any]] = []
        with EvidenceStore(self.database_path) as store:
            for split in _SPLIT_RULES:
                for item in queue[split]:
                    source = sources[str(item["source_id"])]
                    passage = store.get_passage(str(item["source_id"]), str(item["reference"]))
                    if passage is None:  # audit already proves this; retain fail-closed behavior.
                        raise ValueError(f"passage disappeared during packet build: {item['item_id']}")
                    category = self._category(item["category"])
                    required_reviewers = (
                        2
                        if split == "preferences"
                        or category in ReviewedDatasetValidator.SENSITIVE_CATEGORIES
                        else 1
                    )
                    packet = {
                        **item,
                        "split": split,
                        "status": "draft_authoring_worksheet",
                        "scope": "scenario_and_evidence_only_not_a_training_candidate",
                        "required_independent_reviewers": required_reviewers,
                        "evidence_snapshot": {
                            "source_id": passage.source_id,
                            "reference": passage.reference,
                            "quotation": passage.text,
                            "language": passage.language,
                            "source_revision": source["revision"],
                            "canonical_source_sha256": source["sha256"],
                            "license_decision_id": source["approval"]["decision_id"],
                            "required_attribution": source["required_attribution"],
                        },
                    }
                    canonical = json.dumps(
                        packet, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    packet["authoring_packet_sha256"] = hashlib.sha256(
                        canonical.encode("utf-8")
                    ).hexdigest()
                    packets.append(packet)
        destination = self.root / output_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        ) as handle:
            for packet in packets:
                handle.write(json.dumps(packet, ensure_ascii=False, sort_keys=True) + "\n")
            temporary = Path(handle.name)
        temporary.replace(destination)
        return {
            "status": "authoring_packets_built",
            "output_path": str(destination),
            "packet_count": len(packets),
            "counts": report.counts,
            "sensitive_counts": report.sensitive_counts,
        }

    def write_cpu_audit_receipt(
        self, output_path: str | Path = "data/audit/pilot_cpu_validation.json"
    ) -> dict[str, Any]:
        from .pilot import PilotWorkflow
        from .pilot_candidates import PilotCandidateWorkflow
        from .review_ledger import ReviewLedgerValidator
        from .reviewers import ReviewerWorkflow

        draft_report = self.audit()
        candidate_report = PilotCandidateWorkflow(self.root).audit()
        reviewer_report = ReviewerWorkflow(self.root).audit_readiness()
        review_report = ReviewLedgerValidator(self.root).validate()
        pilot_report = PilotWorkflow(self.root).readiness()
        authoring_packet_path = self.root / "data/pilot/authoring_packets.jsonl"
        candidate_packet_path = self.root / "data/pilot/candidate_review_packets.jsonl"
        ready = (
            draft_report.passed
            and candidate_report.passed
            and reviewer_report.passed
            and review_report.passed
            and pilot_report.ready
        )
        receipt: dict[str, Any] = {
            "schema_version": "1.0",
            "generated_at": datetime.now(UTC).isoformat(),
            "status": "ready" if ready else "blocked",
            "git_commit": self._git_commit(),
            "inputs": {
                "draft_queue_sha256": file_sha256(self.queue_path),
                "source_registry_sha256": file_sha256(
                    self.root / "configs/data/source_registry.json"
                ),
                "reviewer_registry_sha256": file_sha256(
                    self.root / "configs/reviewers.json"
                ),
                "evidence_database_sha256": file_sha256(self.database_path),
                "authoring_packets_sha256": (
                    file_sha256(authoring_packet_path)
                    if authoring_packet_path.is_file()
                    else None
                ),
                "candidate_review_packets_sha256": (
                    file_sha256(candidate_packet_path)
                    if candidate_packet_path.is_file()
                    else None
                ),
            },
            "draft_audit": draft_report.to_dict(),
            "candidate_audit": candidate_report.to_dict(),
            "reviewer_readiness": reviewer_report.to_dict(),
            "review_ledger": review_report.to_dict(),
            "pilot_preflight": pilot_report.to_dict(),
        }
        canonical = json.dumps(
            receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        receipt["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        destination = self.root / output_path
        self._write_json(destination, receipt)
        return {
            "status": receipt["status"],
            "output_path": str(destination),
            "receipt_sha256": receipt["receipt_sha256"],
        }

    @staticmethod
    def _category(value: object) -> str:
        return re.sub(r"[\s-]+", "_", str(value).strip().casefold())

    def _git_commit(self) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=self.root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            temporary = Path(handle.name)
        temporary.replace(path)
