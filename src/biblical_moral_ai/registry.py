"""Load and validate versioned policy registries without optional dependencies."""

from __future__ import annotations

import json
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
