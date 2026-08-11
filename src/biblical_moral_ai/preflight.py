"""Repository validation and fail-closed training readiness checks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .arithmetic import verify_equation
from .canon import CanonRegistry
from .evidence_store import file_sha256
from .registry import RegistryError, load_commandment_rules, load_json, load_prophetic_rules


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    name: str
    passed: bool
    detail: str
    blocking: bool = True


@dataclass(frozen=True, slots=True)
class PreflightReport:
    ready: bool
    checks: tuple[PreflightCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_REQUIRED_FILES = (
    "MORAL_CONSTITUTION.md",
    "HERMENEUTICS_POLICY.md",
    "THEOLOGY_POLICY.md",
    "SAFETY_POLICY.md",
    "DATA_CARD.md",
    "MODEL_CARD.md",
    "RISK_REGISTER.md",
    "LICENSES.md",
    "PROPHETIC_RULE_REGISTRY.yaml",
    "configs/commandments.json",
    "configs/canon.json",
    "configs/data/source_registry.json",
    "configs/data/source_packages.json",
    "configs/data/lexicon_packages.json",
    "configs/training/apertus_8b_qlora.json",
    "configs/training/apertus_8b_dpo.json",
    "data/registry/dataset_manifest.json",
    "evals/sealed/manifest.json",
)

_REQUIRED_SOURCE_ROLES = {
    "primary_english_display",
    "operational_hebrew_aramaic",
    "controlling_new_testament_greek",
    "operational_greek_morphology",
    "operational_hebrew_aramaic_lexicon",
    "operational_koine_greek_lexicon",
}


class ProjectPreflight:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def validate_structure(self) -> PreflightReport:
        checks: list[PreflightCheck] = []
        missing = [path for path in _REQUIRED_FILES if not (self.root / path).is_file()]
        checks.append(
            PreflightCheck(
                "required_files",
                not missing,
                "all required governance and configuration files exist"
                if not missing
                else f"missing: {missing}",
            )
        )

        try:
            rules = load_commandment_rules(self.root / "configs/commandments.json")
            checks.append(
                PreflightCheck("commandment_registry", len(rules) == 10, "commandments 1-10 loaded")
            )
        except (RegistryError, ValueError) as exc:
            checks.append(PreflightCheck("commandment_registry", False, str(exc)))

        try:
            canon = CanonRegistry.load(self.root / "configs/canon.json")
            checks.append(
                PreflightCheck(
                    "canon_registry",
                    len(canon.books) == 66,
                    "66 books loaded: 39 Old Testament and 27 New Testament",
                )
            )
        except (RegistryError, ValueError, TypeError, KeyError) as exc:
            checks.append(PreflightCheck("canon_registry", False, str(exc)))

        try:
            prophetic = load_prophetic_rules(self.root / "PROPHETIC_RULE_REGISTRY.yaml")
            equations = [
                vector for rule in prophetic.values() for vector in rule.get("test_vectors", [])
            ]
            arithmetic_passed = bool(equations) and all(verify_equation(item) for item in equations)
            checks.append(
                PreflightCheck(
                    "prophetic_registry",
                    arithmetic_passed,
                    f"{len(prophetic)} rules and {len(equations)} deterministic test vectors validated",
                )
            )
        except (RegistryError, ValueError) as exc:
            checks.append(PreflightCheck("prophetic_registry", False, str(exc)))

        json_paths = [
            *self.root.joinpath("configs").rglob("*.json"),
            *self.root.joinpath("schemas").rglob("*.json"),
            *self.root.joinpath("data", "registry").rglob("*.json"),
            *self.root.joinpath("evals").rglob("*.json"),
        ]
        json_errors: list[str] = []
        for path in json_paths:
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                json_errors.append(f"{path.relative_to(self.root)}: {exc}")
        checks.append(
            PreflightCheck(
                "json_artifacts",
                not json_errors,
                f"{len(json_paths)} JSON artifacts parsed"
                if not json_errors
                else "; ".join(json_errors),
            )
        )

        try:
            public_cases = load_json(self.root / "evals/public/commandment_cases.json")["cases"]
            covered = {number for case in public_cases for number in case["commandments"]}
            checks.append(
                PreflightCheck(
                    "public_commandment_coverage",
                    covered == set(range(1, 11)),
                    f"covered commandments: {sorted(covered)}",
                )
            )
        except (RegistryError, KeyError, TypeError) as exc:
            checks.append(PreflightCheck("public_commandment_coverage", False, str(exc)))

        return PreflightReport(
            ready=all(check.passed for check in checks if check.blocking),
            checks=tuple(checks),
        )

    def training_readiness(self, *, stage: str = "all") -> PreflightReport:
        if stage not in {"all", "sft", "dpo"}:
            raise ValueError(f"unsupported training stage: {stage}")
        checks = list(self.validate_structure().checks)

        try:
            sources = load_json(self.root / "configs/data/source_registry.json")["sources"]
            required_roles = _REQUIRED_SOURCE_ROLES
            approved_roles = {
                source["role"]
                for source in sources
                if source.get("status") == "approved"
                and source.get("revision")
                and source.get("sha256")
            }
            missing_roles = sorted(required_roles - approved_roles)
            checks.append(
                PreflightCheck(
                    "approved_textual_sources",
                    not missing_roles,
                    "all required corpora are approved and pinned"
                    if not missing_roles
                    else f"unapproved or unpinned roles: {missing_roles}",
                )
            )
            source_packages = load_json(
                self.root / "configs/data/source_packages.json"
            )["packages"]
            lexicon_packages = load_json(
                self.root / "configs/data/lexicon_packages.json"
            )["packages"]
            locked = {item["source_id"]: item for item in source_packages + lexicon_packages}
            approved = [item for item in sources if item.get("role") in required_roles]
            lock_errors = []
            for source in approved:
                package = locked.get(source["source_id"])
                if package is None:
                    lock_errors.append(f"{source['source_id']}: lock missing")
                    continue
                if package.get("revision") != source.get("revision"):
                    lock_errors.append(f"{source['source_id']}: revision mismatch")
                if package.get("canonical_artifact_sha256") != source.get("sha256"):
                    lock_errors.append(f"{source['source_id']}: canonical digest mismatch")
            checks.append(
                PreflightCheck(
                    "source_package_locks",
                    not lock_errors and len(approved) == len(required_roles),
                    "registry approvals match immutable package locks"
                    if not lock_errors
                    else "; ".join(lock_errors),
                )
            )
        except (RegistryError, KeyError, TypeError) as exc:
            checks.append(PreflightCheck("approved_textual_sources", False, str(exc)))

        try:
            evidence_manifest_path = self.root / "data/index/evidence_manifest.json"
            evidence_manifest = load_json(evidence_manifest_path)
            database_path = self.root / "data/index/biblical_evidence.sqlite3"
            corpus_path = self.root / "data/index/citation_corpus.json"
            inventory_roles = {item["role"] for item in evidence_manifest["inventory"]}
            evidence_ready = (
                inventory_roles >= _REQUIRED_SOURCE_ROLES
                and database_path.is_file()
                and corpus_path.is_file()
                and file_sha256(database_path) == evidence_manifest["database_sha256"]
                and file_sha256(corpus_path) == evidence_manifest["citation_corpus_sha256"]
            )
            checks.append(
                PreflightCheck(
                    "built_evidence_store",
                    evidence_ready,
                    "all pinned Scripture and lexicons are imported with matching digests"
                    if evidence_ready
                    else "run build-evidence and preserve its matching manifest",
                )
            )
        except (RegistryError, KeyError, TypeError, OSError) as exc:
            checks.append(PreflightCheck("built_evidence_store", False, str(exc)))

        try:
            manifest = load_json(self.root / "data/registry/dataset_manifest.json")
            required_splits = {
                "sft": ("sft_train",),
                "dpo": ("preference_train",),
                "all": ("sft_train", "preference_train"),
            }[stage]
            split_checks = []
            for split_name in required_splits:
                split = manifest["splits"][split_name]
                path = self.root / split["path"]
                passed = (
                    split["accepted_count"] >= split["target_minimum"]
                    and path.is_file()
                    and bool(split["sha256"])
                    and file_sha256(path) == split["sha256"]
                )
                split_checks.append(
                    (split_name, passed, split["accepted_count"], split["target_minimum"])
                )
            checks.append(
                PreflightCheck(
                    "reviewed_training_data",
                    all(item[1] for item in split_checks),
                    ", ".join(
                        f"{name}={count}/{target}" for name, _, count, target in split_checks
                    ),
                )
            )
        except (RegistryError, KeyError, TypeError) as exc:
            checks.append(PreflightCheck("reviewed_training_data", False, str(exc)))

        try:
            sft_config = load_json(self.root / "configs/training/apertus_8b_qlora.json")
            model = sft_config["model"]
            pinned = bool(
                model.get("model_id") and model.get("revision") and model.get("reported_license")
            )
            checks.append(
                PreflightCheck(
                    "base_model_pinned", pinned, f"{model.get('model_id')}@{model.get('revision')}"
                )
            )
        except (RegistryError, KeyError, TypeError) as exc:
            checks.append(PreflightCheck("base_model_pinned", False, str(exc)))

        return PreflightReport(
            ready=all(check.passed for check in checks if check.blocking),
            checks=tuple(checks),
        )
