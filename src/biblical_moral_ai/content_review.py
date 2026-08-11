"""Fail-closed clarity, precision, and contradiction review."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from .decisions import strongest_decision
from .registry import load_deception_taxonomy, load_json
from .schemas import (
    AssessmentVerdict,
    Confidence,
    IssueSeverity,
    MoralAnswer,
    PipelineDecision,
    ReviewStatus,
    VerificationIssue,
    VerificationReport,
)


@dataclass(frozen=True, slots=True)
class RepositoryContentIssue:
    code: str
    message: str
    suggested_correction: str
    severity: IssueSeverity = IssueSeverity.CRITICAL
    path: str = ""
    line: int | None = None


@dataclass(frozen=True, slots=True)
class RepositoryContentReport:
    passed: bool
    files_checked: int
    invariants_checked: int
    issues: tuple[RepositoryContentIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SENTENCE = re.compile(r"[^.!?]+[.!?]?")
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")
_NEGATION = frozenset({"not", "no", "never", "cannot", "neither", "nor"})
_SIGNATURE_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "to",
        "of",
        "and",
        "or",
        "that",
        "this",
    }
)
_MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank JSONL record at {path}:{line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
        records.append(value)
    return records


class AdvancedContentReviewer:
    """Review one structured answer without inventing a replacement conclusion."""

    def __init__(self, rules: Mapping[str, Any]) -> None:
        self.max_sentence_words = int(rules["max_sentence_words"])
        self.similarity_threshold = float(rules["similarity_threshold"])
        self.unresolved_markers = tuple(str(item) for item in rules["unresolved_markers"])
        self.vague_terms = tuple(str(item) for item in rules["vague_terms"])
        self.absolute_terms = tuple(
            str(item) for item in rules["absolute_certainty_terms"]
        )
        self.language_claim_patterns = tuple(
            re.compile(str(item), re.IGNORECASE)
            for item in rules["source_language_claim_patterns"]
        )
        self.language_disclaimers = tuple(
            str(item).casefold() for item in rules["source_language_disclaimer_patterns"]
        )
        self.contradiction_pairs = tuple(
            (
                str(item["rule_id"]),
                re.compile(str(item["left_pattern"]), re.IGNORECASE),
                re.compile(str(item["right_pattern"]), re.IGNORECASE),
                str(item["message"]),
            )
            for item in rules["contradiction_pairs"]
        )

    def check(self, answer: MoralAnswer) -> VerificationReport:
        issues: list[VerificationIssue] = []
        fields = tuple(self._review_fields(answer))
        content_text = " ".join(value for _, value in fields)

        issues.extend(self._clarity_issues(fields))
        issues.extend(self._duplicate_issues(answer))
        issues.extend(self._proposition_contradictions(fields))
        issues.extend(self._configured_contradictions(content_text))
        issues.extend(self._confidence_issues(answer, content_text))
        issues.extend(self._source_language_issues(answer, content_text))
        issues.extend(self._assessment_issues(answer))

        return VerificationReport(
            decision=strongest_decision(issues),
            issues=tuple(issues),
            checks={
                "content_has_no_contradictions": not any(
                    issue.code.startswith("CONTENT_CONTRADICTION") for issue in issues
                ),
                "content_is_clear_and_precise": not any(
                    issue.code.startswith(("CONTENT_CLARITY", "CONTENT_PRECISION"))
                    for issue in issues
                ),
                "content_confidence_is_consistent": not any(
                    issue.code.startswith("CONTENT_CONFIDENCE") for issue in issues
                ),
            },
        )

    @staticmethod
    def _review_fields(answer: MoralAnswer) -> Iterable[tuple[str, str]]:
        yield "conclusion", answer.conclusion
        for field_name in (
            "known_facts",
            "missing_information",
            "moral_duties",
            "potential_harms",
            "alternatives",
            "practical_options",
            "human_referral",
        ):
            for index, value in enumerate(getattr(answer, field_name)):
                yield f"{field_name}[{index}]", value
        for index, item in enumerate(answer.evidence):
            yield f"evidence[{index}].claim", item.claim
            if item.immediate_context:
                yield f"evidence[{index}].immediate_context", item.immediate_context
            if item.language_notes:
                yield f"evidence[{index}].language_notes", item.language_notes
            for assumption_index, value in enumerate(item.assumptions):
                yield f"evidence[{index}].assumptions[{assumption_index}]", value
        for index, item in enumerate(answer.commandment_assessments):
            yield f"commandment_assessments[{index}].rationale", item.rationale
            if item.remediation:
                yield f"commandment_assessments[{index}].remediation", item.remediation

    def _clarity_issues(
        self, fields: Iterable[tuple[str, str]]
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []
        for path, value in fields:
            lowered = value.casefold()
            for marker in self.unresolved_markers:
                if marker.casefold() in lowered:
                    issues.append(
                        self._issue(
                            "CONTENT_CLARITY_UNRESOLVED_MARKER",
                            f"Unresolved marker {marker!r} remains in {path}.",
                            path,
                            "Replace the marker with reviewed content or remove the incomplete statement.",
                        )
                    )
            for term in self.vague_terms:
                if re.search(rf"(?<!\w){re.escape(term.casefold())}(?!\w)", lowered):
                    issues.append(
                        self._issue(
                            "CONTENT_CLARITY_VAGUE_TERM",
                            f"Vague term {term!r} appears in {path}.",
                            path,
                            "Name the actor, action, evidence, scope, or condition explicitly.",
                        )
                    )
            if path.startswith(("conclusion", "moral_duties", "alternatives", "practical_options")):
                for sentence in _SENTENCE.findall(value):
                    word_count = len(_WORD.findall(sentence))
                    if word_count > self.max_sentence_words:
                        issues.append(
                            self._issue(
                                "CONTENT_CLARITY_SENTENCE_TOO_LONG",
                                f"A {word_count}-word sentence in {path} exceeds the {self.max_sentence_words}-word clarity limit.",
                                path,
                                "Split the sentence into shorter statements with one principal claim each.",
                            )
                        )
        return issues

    def _duplicate_issues(self, answer: MoralAnswer) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []
        for field_name in (
            "known_facts",
            "missing_information",
            "moral_duties",
            "affected_people",
            "potential_harms",
            "alternatives",
            "practical_options",
            "human_referral",
        ):
            values = getattr(answer, field_name)
            normalized = [self._normalize(value) for value in values]
            if len(normalized) != len(set(normalized)):
                issues.append(
                    self._issue(
                        "CONTENT_PRECISION_DUPLICATE_ITEM",
                        f"{field_name} contains duplicate content.",
                        field_name,
                        "Keep one precise instance of each distinct point.",
                    )
                )
        return issues

    def _proposition_contradictions(
        self, fields: Iterable[tuple[str, str]]
    ) -> list[VerificationIssue]:
        propositions: list[tuple[str, str, bool]] = []
        for path, value in fields:
            if path.startswith(("commandment_assessments", "evidence")):
                continue
            for sentence in _SENTENCE.findall(value):
                signature, negative = self._proposition_signature(sentence)
                if len(signature.split()) >= 3:
                    propositions.append((path, signature, negative))

        issues: list[VerificationIssue] = []
        seen: set[tuple[str, str]] = set()
        for index, (left_path, left, left_negative) in enumerate(propositions):
            for right_path, right, right_negative in propositions[index + 1 :]:
                if left_path == right_path or left_negative == right_negative:
                    continue
                similarity = SequenceMatcher(None, left, right).ratio()
                if similarity < self.similarity_threshold:
                    continue
                pair = tuple(sorted((left_path, right_path)))
                if pair in seen:
                    continue
                seen.add(pair)
                issues.append(
                    self._issue(
                        "CONTENT_CONTRADICTION_NEGATED_PROPOSITION",
                        f"Opposite versions of the same proposition appear in {left_path} and {right_path}.",
                        f"{left_path};{right_path}",
                        "Resolve which proposition is supported, cite the evidence, and remove or qualify the other.",
                    )
                )
        return issues

    def _configured_contradictions(self, text: str) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []
        for rule_id, left, right, message in self.contradiction_pairs:
            if left.search(text) and right.search(text):
                issues.append(
                    self._issue(
                        f"CONTENT_CONTRADICTION_{rule_id.upper()}",
                        message,
                        "answer",
                        "Retain the evidence-supported claim, qualify uncertainty, and remove the incompatible claim.",
                    )
                )
        return issues

    def _confidence_issues(
        self, answer: MoralAnswer, content_text: str
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []
        pending = [
            index
            for index, item in enumerate(answer.evidence)
            if item.reviewer_status in {ReviewStatus.UNREVIEWED, ReviewStatus.PENDING}
        ]
        if answer.confidence is Confidence.HIGH and pending:
            issues.append(
                self._issue(
                    "CONTENT_CONFIDENCE_HIGH_WITH_UNREVIEWED_EVIDENCE",
                    f"High confidence relies on unreviewed or pending evidence at indexes {pending}.",
                    "confidence",
                    "Complete evidence review or reduce confidence before release.",
                    decision=PipelineDecision.ESCALATE,
                )
            )
        if answer.missing_information:
            lowered = content_text.casefold()
            for term in self.absolute_terms:
                if term.casefold() in lowered:
                    issues.append(
                        self._issue(
                            "CONTENT_CONFIDENCE_ABSOLUTE_WITH_MISSING_INFORMATION",
                            f"Absolute term {term!r} conflicts with acknowledged missing information.",
                            "answer",
                            "Bound the claim, state the controlling assumptions, and use calibrated confidence.",
                        )
                    )
        return issues

    def _source_language_issues(
        self, answer: MoralAnswer, content_text: str
    ) -> list[VerificationIssue]:
        if not any(pattern.search(content_text) for pattern in self.language_claim_patterns):
            return []
        usable_language_support = any(
            item.language_notes.strip()
            and not any(
                disclaimer in item.language_notes.casefold()
                for disclaimer in self.language_disclaimers
            )
            for item in answer.evidence
        )
        if usable_language_support:
            return []
        return [
            self._issue(
                "CONTENT_PRECISION_UNSUPPORTED_SOURCE_LANGUAGE_CLAIM",
                "A Hebrew, Aramaic, or Greek conclusion is asserted without reviewed source-language support.",
                "answer",
                "Retrieve the approved language evidence, record the lemma and context, and obtain qualified review; otherwise remove or qualify the claim.",
                decision=PipelineDecision.ESCALATE,
            )
        ]

    def _assessment_issues(self, answer: MoralAnswer) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []
        for index, item in enumerate(answer.commandment_assessments):
            lowered = item.rationale.casefold()
            path = f"commandment_assessments[{index}].rationale"
            if item.verdict is AssessmentVerdict.COMPLIANT and re.search(
                r"(?<!does not )\b(?:violates|violation|breaches|conflicts with)\b", lowered
            ):
                issues.append(
                    self._issue(
                        "CONTENT_CONTRADICTION_ASSESSMENT_VERDICT",
                        f"Commandment {item.commandment} is marked compliant but its rationale describes a violation.",
                        path,
                        "Correct the verdict or rewrite the rationale to match the reviewed assessment.",
                    )
                )
            if item.verdict is AssessmentVerdict.VIOLATION and re.search(
                r"\b(?:complies|compliant|does not violate)\b", lowered
            ):
                issues.append(
                    self._issue(
                        "CONTENT_CONTRADICTION_ASSESSMENT_VERDICT",
                        f"Commandment {item.commandment} is marked violation but its rationale describes compliance.",
                        path,
                        "Correct the verdict or rewrite the rationale to match the reviewed assessment.",
                    )
                )
        return issues

    @staticmethod
    def _issue(
        code: str,
        message: str,
        field_path: str,
        suggestion: str,
        *,
        decision: PipelineDecision = PipelineDecision.CORRECT,
    ) -> VerificationIssue:
        return VerificationIssue(
            code=code,
            message=f"{message} Suggested correction: {suggestion}",
            decision=decision,
            severity=IssueSeverity.CRITICAL,
            field_path=field_path,
        )

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(_WORD.findall(value.casefold()))

    @staticmethod
    def _proposition_signature(value: str) -> tuple[str, bool]:
        normalized = value.casefold().replace("can't", "cannot").replace("won't", "will not")
        tokens = _WORD.findall(normalized)
        negative = any(token in _NEGATION or token.endswith("n't") for token in tokens)
        signature = " ".join(
            token
            for token in tokens
            if token not in _NEGATION
            and not token.endswith("n't")
            and token not in _SIGNATURE_STOP_WORDS
        )
        return signature, negative


class RepositoryContentReviewer:
    """Audit governed repository content against derived, current invariants."""

    def __init__(self, root: str | Path, rules: Mapping[str, Any]) -> None:
        self.root = Path(root).resolve()
        self.rules = rules
        self.invariants = rules["repository_invariants"]
        self.unresolved_markers = tuple(str(item) for item in rules["unresolved_markers"])

    def audit(self) -> RepositoryContentReport:
        issues: list[RepositoryContentIssue] = []
        files = tuple(self._governed_markdown_files())
        issues.extend(self._review_markdown(files))
        invariant_issues, invariant_count = self._review_invariants()
        issues.extend(invariant_issues)
        return RepositoryContentReport(
            passed=not issues,
            files_checked=len(files),
            invariants_checked=invariant_count,
            issues=tuple(issues),
        )

    def _governed_markdown_files(self) -> Iterable[Path]:
        roots = [self.root, self.root / "docs", self.root / "data" / "pilot" / "candidates"]
        seen: set[Path] = set()
        for base in roots:
            if not base.exists():
                continue
            pattern = "*.md" if base == self.root else "**/*.md"
            for path in base.glob(pattern):
                if path.is_file() and path not in seen:
                    seen.add(path)
                    yield path
        system_prompt = self.root / "configs" / "inference" / "system_prompt.txt"
        if system_prompt.is_file() and system_prompt not in seen:
            yield system_prompt

    def _review_markdown(self, files: Iterable[Path]) -> list[RepositoryContentIssue]:
        issues: list[RepositoryContentIssue] = []
        for path in files:
            text = path.read_text(encoding="utf-8")
            relative = path.relative_to(self.root).as_posix()
            for line_number, line in enumerate(text.splitlines(), 1):
                lowered = line.casefold()
                for marker in self.unresolved_markers:
                    if marker.casefold() in lowered:
                        issues.append(
                            RepositoryContentIssue(
                                "REPO_UNRESOLVED_MARKER",
                                f"Unresolved marker {marker!r} appears in governed content.",
                                "Replace it with reviewed content or clearly label the section as a non-authoritative template.",
                                path=relative,
                                line=line_number,
                            )
                        )
                for match in _MARKDOWN_LINK.finditer(line):
                    raw_target = match.group(1).strip()
                    if raw_target.startswith("<") and ">" in raw_target:
                        target = raw_target[1 : raw_target.index(">")]
                    else:
                        target = raw_target.split(maxsplit=1)[0]
                    if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                        continue
                    target_path = unquote(target.split("#", 1)[0])
                    if not target_path:
                        continue
                    resolved = (path.parent / target_path).resolve()
                    try:
                        resolved.relative_to(self.root)
                    except ValueError:
                        continue
                    if not resolved.exists():
                        issues.append(
                            RepositoryContentIssue(
                                "REPO_BROKEN_LOCAL_LINK",
                                f"Local Markdown link target does not exist: {target}",
                                "Correct the relative path or remove the stale link.",
                                path=relative,
                                line=line_number,
                            )
                        )
        return issues

    def _review_invariants(self) -> tuple[list[RepositoryContentIssue], int]:
        issues: list[RepositoryContentIssue] = []
        checks = 0

        canon = load_json(self.root / "configs/canon.json")
        actual_canon = len(canon.get("books", []))
        expected_canon = int(self.invariants["canon_book_count"])
        checks += 1
        if actual_canon != expected_canon:
            issues.append(
                self._invariant_issue(
                    "REPO_CANON_COUNT_CONTRADICTION",
                    f"Canon registry has {actual_canon} books but policy requires {expected_canon}.",
                    "configs/canon.json",
                    "Reconcile the canon registry and every documented canon claim through policy review.",
                )
            )

        queue = load_json(self.root / "configs/pilot/draft_scenarios.json")
        expected_counts = self.invariants["pilot_counts"]
        for split, expected in expected_counts.items():
            checks += 1
            actual = len(queue.get(split, []))
            if actual != int(expected):
                issues.append(
                    self._invariant_issue(
                        "REPO_PILOT_QUEUE_COUNT_CONTRADICTION",
                        f"Pilot queue {split} has {actual} records but the governed target is {expected}.",
                        "configs/pilot/draft_scenarios.json",
                        "Reconcile the queue, manifest, documentation, and review requirements together.",
                    )
                )
            candidate_path = self.root / "data" / "pilot" / "candidates" / f"{split}.jsonl"
            checks += 1
            candidate_count = len(_read_jsonl(candidate_path)) if candidate_path.is_file() else 0
            if candidate_count != actual:
                issues.append(
                    self._invariant_issue(
                        "REPO_CANDIDATE_COUNT_CONTRADICTION",
                        f"Candidate split {split} has {candidate_count} records but its queue has {actual}.",
                        candidate_path.relative_to(self.root).as_posix(),
                        "Regenerate untouched candidates or complete a reviewed migration; never silently drop records.",
                    )
                )

        taxonomy = load_deception_taxonomy(self.root / "configs/deception_taxonomy.json")
        checks += 1
        if len(taxonomy) < int(self.invariants["minimum_deception_types"]):
            issues.append(
                self._invariant_issue(
                    "REPO_DECEPTION_COVERAGE_CONTRADICTION",
                    f"Deception taxonomy has only {len(taxonomy)} operational types.",
                    "configs/deception_taxonomy.json",
                    "Restore comprehensive coverage and its deterministic refusal vectors.",
                )
            )
        catch_all = str(self.invariants["required_deception_catch_all"])
        checks += 1
        if catch_all not in taxonomy:
            issues.append(
                self._invariant_issue(
                    "REPO_DECEPTION_CATCH_ALL_MISSING",
                    f"Required deception catch-all {catch_all!r} is missing.",
                    "configs/deception_taxonomy.json",
                    "Restore the catch-all so novel intentional false impressions fail closed.",
                )
            )

        source_registry = load_json(self.root / "configs/data/source_registry.json")
        approved_roles = {
            str(item.get("role"))
            for item in source_registry.get("sources", [])
            if item.get("status") == "approved"
        }
        required_roles = set(self.invariants["required_source_roles"])
        checks += 1
        missing_roles = sorted(required_roles - approved_roles)
        if missing_roles:
            issues.append(
                self._invariant_issue(
                    "REPO_SOURCE_ROLE_CONTRADICTION",
                    f"Approved source registry is missing required roles: {missing_roles}.",
                    "configs/data/source_registry.json",
                    "Approve and pin the required sources or correct the governed source policy.",
                )
            )

        documentation_checks = (
            (
                "README.md",
                rf"{len(taxonomy)} operational deception categories",
                "REPO_DOCUMENTED_DECEPTION_COUNT_STALE",
            ),
            (
                "IMPLEMENTATION_STATUS.md",
                rf"{len(taxonomy)}-category deception taxonomy",
                "REPO_DOCUMENTED_DECEPTION_COUNT_STALE",
            ),
            (
                "docs/PILOT_REVIEW_WORKFLOW.md",
                rf"{expected_counts['sft']} SFT, {expected_counts['preferences']} preference, and {expected_counts['evals']} evaluation",
                "REPO_DOCUMENTED_PILOT_COUNT_STALE",
            ),
        )
        for relative, expected_text, code in documentation_checks:
            checks += 1
            document = (self.root / relative).read_text(encoding="utf-8")
            if expected_text not in document:
                issues.append(
                    self._invariant_issue(
                        code,
                        f"Documented invariant is missing or stale: {expected_text!r}.",
                        relative,
                        "Update the prose from the machine-readable source in the same change.",
                    )
                )

        return issues, checks

    @staticmethod
    def _invariant_issue(
        code: str, message: str, path: str, suggestion: str
    ) -> RepositoryContentIssue:
        return RepositoryContentIssue(code, message, suggestion, path=path)
