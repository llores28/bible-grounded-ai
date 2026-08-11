"""Load and validate versioned policy registries without optional dependencies."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .schemas import CommandmentRule


class RegistryError(ValueError):
    """Raised when a governance registry is missing or internally inconsistent."""


def load_json(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path)
    try:
        with registry_path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise RegistryError(f"registry does not exist: {registry_path}") from exc
    except json.JSONDecodeError as exc:
        raise RegistryError(
            f"registry is not valid JSON-compatible YAML: {registry_path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise RegistryError(f"registry root must be an object: {registry_path}")
    return value


def load_commandment_rules(path: str | Path) -> dict[int, CommandmentRule]:
    payload = load_json(path)
    if payload.get("schema_version") != "1.0":
        raise RegistryError("unsupported commandment registry schema_version")
    raw_rules = payload.get("rules")
    if not isinstance(raw_rules, list):
        raise RegistryError("commandment registry rules must be a list")
    rules = [CommandmentRule.from_dict(item) for item in raw_rules]
    by_number = {rule.number: rule for rule in rules}
    if len(rules) != 10 or set(by_number) != set(range(1, 11)):
        raise RegistryError("commandment registry must define each commandment exactly once")
    return by_number


def load_deception_taxonomy(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    if payload.get("schema_version") != "1.0":
        raise RegistryError("unsupported deception taxonomy schema_version")
    raw_types = payload.get("types")
    if not isinstance(raw_types, list) or len(raw_types) < 20:
        raise RegistryError("deception taxonomy must define at least 20 operational types")

    by_id: dict[str, dict[str, Any]] = {}
    required_lists = ("recommendation_terms", "safe_alternatives", "anchor_passages")
    for item in raw_types:
        if not isinstance(item, dict):
            raise RegistryError("every deception taxonomy entry must be an object")
        type_id = item.get("type_id")
        if not isinstance(type_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]+", type_id):
            raise RegistryError(f"invalid deception type_id: {type_id!r}")
        if type_id in by_id:
            raise RegistryError(f"duplicate deception type_id: {type_id}")
        for field in ("label", "definition", "unsafe_test_phrase"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise RegistryError(f"{type_id} requires non-empty {field}")
        for field in required_lists:
            values = item.get(field)
            if not isinstance(values, list) or not values or not all(
                isinstance(value, str) and value.strip() for value in values
            ):
                raise RegistryError(f"{type_id} requires non-empty {field}")
        if not isinstance(item.get("high_impact"), bool):
            raise RegistryError(f"{type_id} requires a boolean high_impact value")
        test_phrase = item["unsafe_test_phrase"].casefold()
        if not any(term.casefold() in test_phrase for term in item["recommendation_terms"]):
            raise RegistryError(
                f"{type_id} unsafe_test_phrase must contain a recommendation term"
            )
        by_id[type_id] = item

    if "other_intentional_false_impression" not in by_id:
        raise RegistryError("deception taxonomy requires the catch-all false-impression type")
    return by_id


def load_content_review_rules(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("schema_version") != "1.0":
        raise RegistryError("unsupported content review rules schema_version")
    max_words = payload.get("max_sentence_words")
    threshold = payload.get("similarity_threshold")
    if not isinstance(max_words, int) or not 20 <= max_words <= 100:
        raise RegistryError("max_sentence_words must be an integer between 20 and 100")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise RegistryError("similarity_threshold must be numeric")
    if not 0.8 <= float(threshold) <= 1.0:
        raise RegistryError("similarity_threshold must be between 0.8 and 1.0")

    for field in (
        "unresolved_markers",
        "vague_terms",
        "absolute_certainty_terms",
        "source_language_claim_patterns",
        "source_language_disclaimer_patterns",
    ):
        values = payload.get(field)
        if not isinstance(values, list) or not values or not all(
            isinstance(value, str) and value.strip() for value in values
        ):
            raise RegistryError(f"content review rules require non-empty {field}")
        if field.endswith("_patterns"):
            for value in values:
                try:
                    re.compile(value, re.IGNORECASE)
                except re.error as exc:
                    raise RegistryError(f"invalid regex in {field}: {value}: {exc}") from exc

    pairs = payload.get("contradiction_pairs")
    if not isinstance(pairs, list) or not pairs:
        raise RegistryError("content review rules require contradiction_pairs")
    seen: set[str] = set()
    for item in pairs:
        if not isinstance(item, dict):
            raise RegistryError("every contradiction pair must be an object")
        rule_id = item.get("rule_id")
        if not isinstance(rule_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]+", rule_id):
            raise RegistryError(f"invalid contradiction rule_id: {rule_id!r}")
        if rule_id in seen:
            raise RegistryError(f"duplicate contradiction rule_id: {rule_id}")
        seen.add(rule_id)
        for field in ("left_pattern", "right_pattern", "message"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise RegistryError(f"{rule_id} requires non-empty {field}")
        for field in ("left_pattern", "right_pattern"):
            try:
                re.compile(item[field], re.IGNORECASE)
            except re.error as exc:
                raise RegistryError(f"invalid regex in {rule_id}.{field}: {exc}") from exc

    invariants = payload.get("repository_invariants")
    if not isinstance(invariants, dict):
        raise RegistryError("content review rules require repository_invariants")
    counts = invariants.get("pilot_counts")
    if not isinstance(counts, dict) or set(counts) != {"sft", "preferences", "evals"}:
        raise RegistryError("repository invariants require exact pilot split counts")
    if not all(isinstance(value, int) and value > 0 for value in counts.values()):
        raise RegistryError("pilot split counts must be positive integers")
    roles = invariants.get("required_source_roles")
    if not isinstance(roles, list) or not roles or len(set(roles)) != len(roles):
        raise RegistryError("required_source_roles must be a non-empty unique list")
    return payload


def load_prophetic_rules(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    if payload.get("schema_version") != "1.0":
        raise RegistryError("unsupported prophetic registry schema_version")
    rules = payload.get("rules")
    if not isinstance(rules, list) or not rules:
        raise RegistryError("prophetic registry must include at least one rule")
    by_id: dict[str, dict[str, Any]] = {}
    allowed_classes = {
        "explicit_text",
        "canonical_synthesis",
        "contextual_inference",
        "named_historical_interpretation",
        "speculative_hypothesis",
    }
    for rule in rules:
        rule_id = rule.get("rule_id")
        if not isinstance(rule_id, str) or not rule_id:
            raise RegistryError("every prophetic rule requires a rule_id")
        if rule_id in by_id:
            raise RegistryError(f"duplicate prophetic rule_id: {rule_id}")
        if rule.get("evidence_class") not in allowed_classes:
            raise RegistryError(f"unsupported prophetic evidence class in {rule_id}")
        if not rule.get("assumptions") or not rule.get("counter_readings"):
            raise RegistryError(f"{rule_id} must preserve assumptions and counter-readings")
        if rule.get("allow_new_anchor") is not False:
            raise RegistryError(f"{rule_id} must prohibit unreviewed historical anchors")
        by_id[rule_id] = rule
    return by_id
