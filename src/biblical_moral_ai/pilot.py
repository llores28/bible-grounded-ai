"""Fail-closed validation and materialization for the small training pilot."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .citation import CitationVerifier
from .dataset import ReviewedDatasetValidator, materialize_sft_record, read_jsonl
from .evidence_store import file_sha256
from .pipeline import InferenceReviewPipeline
from .policy import CommandmentPolicyEngine
from .preflight import PreflightCheck, PreflightReport, ProjectPreflight
from .registry import load_commandment_rules, load_json


class PilotWorkflow:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def readiness(self) -> PreflightReport:
        checks: list[PreflightCheck] = []
        production = ProjectPreflight(self.root).training_readiness(stage="sft")
        for check in production.checks:
            if check.name in {
                "required_files",
                "commandment_registry",
                "canon_registry",
                "prophetic_registry",
                "json_artifacts",
                "approved_textual_sources",
                "source_package_locks",
                "built_evidence_store",
                "base_model_pinned",
            }:
                checks.append(check)

        try:
            reviewer_registry = load_json(self.root / "configs/reviewers.json")
            active = [
                item
                for item in reviewer_registry["reviewers"]
                if item.get("status") == "active"
                and item.get("affiliations_disclosed") is True
            ]
            checks.append(
                PreflightCheck(
                    "pilot_reviewers",
                    len(active) >= 2,
                    f"active reviewers with disclosed affiliations: {len(active)}/2",
                )
            )
        except (KeyError, TypeError, ValueError, OSError) as exc:
            reviewer_registry = {"reviewers": []}
            checks.append(PreflightCheck("pilot_reviewers", False, str(exc)))

        try:
            corpus_payload = load_json(self.root / "data/index/citation_corpus.json")
            corpora = corpus_payload.get("sources", corpus_payload)
            pipeline = InferenceReviewPipeline(
                commandment_policy=CommandmentPolicyEngine(
                    load_commandment_rules(self.root / "configs/commandments.json")
                ),
                citation_verifier=CitationVerifier(corpora),
                organizational_source_ids=corpus_payload.get(
                    "organizational_source_ids", []
                ),
            )
            source_registry = load_json(self.root / "configs/data/source_registry.json")
            validator = ReviewedDatasetValidator(
                pipeline,
                source_registry=source_registry,
                reviewer_registry=reviewer_registry,
            )
            manifest = load_json(self.root / "data/registry/pilot_manifest.json")
            split_methods = {
                "sft_pilot": validator.validate_sft,
                "preference_pilot": validator.validate_preferences,
                "evaluation_pilot": validator.validate_evals,
            }
            sensitive_found: set[str] = set()
            for split_name, method in split_methods.items():
                split = manifest["splits"][split_name]
                path = self.root / split["path"]
                records = read_jsonl(path) if path.is_file() else []
                report = method(records)
                sensitive_found.update(
                    str(record.get("category", "")).casefold()
                    for record in records
                    if str(record.get("category", "")).casefold()
                    in ReviewedDatasetValidator.SENSITIVE_CATEGORIES
                )
                digest_matches = (
                    path.is_file()
                    and bool(split.get("sha256"))
                    and file_sha256(path) == split["sha256"]
                )
                count_matches = (
                    report.accepted == split["accepted_count"]
                    and report.accepted >= split["target_minimum"]
                )
                detail = (
                    f"accepted={report.accepted}/{split['target_minimum']}; "
                    f"rejected={report.rejected}; digest_match={digest_matches}"
                )
                if report.issues:
                    detail += f"; first_issue={report.issues[0].code}"
                checks.append(
                    PreflightCheck(
                        f"pilot_{split_name}",
                        report.passed and count_matches and digest_matches,
                        detail,
                    )
                )
            missing_sensitive = sorted(
                ReviewedDatasetValidator.SENSITIVE_CATEGORIES - sensitive_found
            )
            checks.append(
                PreflightCheck(
                    "pilot_sensitive_coverage",
                    not missing_sensitive,
                    "all sensitive categories represented with enforced dual review"
                    if not missing_sensitive
                    else f"missing categories: {missing_sensitive}",
                )
            )
        except (KeyError, TypeError, ValueError, OSError) as exc:
            checks.append(PreflightCheck("pilot_datasets", False, str(exc)))

        return PreflightReport(
            ready=all(check.passed for check in checks if check.blocking),
            checks=tuple(checks),
        )

    def materialize(self) -> dict[str, str]:
        readiness = self.readiness()
        if not readiness.ready:
            failed = [check.detail for check in readiness.checks if not check.passed]
            raise ValueError("pilot preflight failed: " + " | ".join(failed))
        manifest = load_json(self.root / "data/registry/pilot_manifest.json")
        system_prompt = (self.root / "configs/inference/system_prompt.txt").read_text(
            encoding="utf-8"
        )
        outputs = {
            "sft": self.root / "data/pilot/materialized_sft.jsonl",
            "eval": self.root / "data/pilot/materialized_eval.jsonl",
        }
        sft = read_jsonl(self.root / manifest["splits"]["sft_pilot"]["path"])
        evaluation = read_jsonl(
            self.root / manifest["splits"]["evaluation_pilot"]["path"]
        )
        self._write_jsonl(
            outputs["sft"],
            [materialize_sft_record(item, system_prompt=system_prompt) for item in sft],
        )
        self._write_jsonl(
            outputs["eval"],
            [
                materialize_sft_record(
                    {**item, "record_id": item["case_id"]},
                    system_prompt=system_prompt,
                )
                for item in evaluation
            ],
        )
        return {name: str(path) for name, path in outputs.items()}

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
