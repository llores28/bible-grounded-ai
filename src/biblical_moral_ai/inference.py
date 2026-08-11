"""Retrieval-first local inference with correction, refusal, and escalation gates."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

from .citation import CitationVerifier
from .content_review import AdvancedContentReviewer
from .decisions import strongest_decision
from .evidence_store import EvidenceStore, Passage
from .pipeline import InferenceReviewPipeline
from .policy import CommandmentPolicyEngine
from .registry import (
    load_commandment_rules,
    load_content_review_rules,
    load_deception_taxonomy,
)
from .render import render_moral_answer
from .schemas import (
    IssueSeverity,
    MoralAnswer,
    PipelineDecision,
    VerificationIssue,
    VerificationReport,
)


class InferenceError(RuntimeError):
    pass


class ChatBackend(Protocol):
    def generate(self, *, system: str, user: str) -> str: ...


@dataclass(frozen=True, slots=True)
class AgentResult:
    decision: PipelineDecision
    report: VerificationReport
    answer: MoralAnswer | None
    delivery_text: str | None
    attempts: int
    retrieved: tuple[Passage, ...]


class LocalChatBackend:
    """Minimal OpenAI-compatible client restricted to loopback by default."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        timeout_seconds: int = 120,
        allow_remote: bool = False,
    ) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("endpoint must use http or https")
        if not allow_remote and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("non-loopback inference endpoints require explicit allow_remote=True")
        self.endpoint = endpoint
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(self, *, system: str, user: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
                "seed": 20260811,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            return str(body["choices"][0]["message"]["content"])
        except (urllib.error.URLError, json.JSONDecodeError, KeyError, IndexError) as exc:
            raise InferenceError(f"local inference request failed: {exc}") from exc


_RETRIEVAL_EXPANSIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\b(?:lie|lying|deceive|fabricat|citation|falsehood)\w*\b", re.IGNORECASE),
        "false witness truth",
    ),
    (
        re.compile(r"\b(?:kill|murder|violence|self-defense|weapon)\w*\b", re.IGNORECASE),
        "murder life",
    ),
    (re.compile(r"\b(?:steal|theft|credential|property)\w*\b", re.IGNORECASE), "steal theft"),
    (
        re.compile(r"\b(?:adultery|affair|infidelity|marriage)\w*\b", re.IGNORECASE),
        "adultery fidelity",
    ),
    (re.compile(r"\b(?:parent|mother|father|honor)\w*\b", re.IGNORECASE), "honour father mother"),
    (re.compile(r"\b(?:covet|greed|envy|jealous)\w*\b", re.IGNORECASE), "covet desire"),
    (re.compile(r"\b(?:sabbath|seventh day)\b", re.IGNORECASE), "sabbath remember holy rest"),
)


class BiblicalMoralAgent:
    def __init__(
        self,
        *,
        root: str | Path,
        store: EvidenceStore,
        backend: ChatBackend,
        system_prompt: str | None = None,
        retrieval_limit: int = 12,
        graph_expansion_limit: int = 8,
        max_corrections: int = 1,
    ) -> None:
        self.root = Path(root).resolve()
        self.store = store
        self.backend = backend
        self.retrieval_limit = retrieval_limit
        self.graph_expansion_limit = graph_expansion_limit
        self.max_corrections = max_corrections
        self.system_prompt = system_prompt or (
            self.root / "configs/inference/system_prompt.txt"
        ).read_text(encoding="utf-8")
        self.commandment_policy = CommandmentPolicyEngine(
            load_commandment_rules(self.root / "configs/commandments.json"),
            load_deception_taxonomy(self.root / "configs/deception_taxonomy.json"),
        )
        self.content_reviewer = AdvancedContentReviewer(
            load_content_review_rules(self.root / "configs/content_review_rules.json")
        )

    def answer(self, request_text: str) -> AgentResult:
        if not request_text.strip():
            raise ValueError("request_text cannot be blank")
        search_text = (
            request_text
            + " "
            + " ".join(
                expansion
                for pattern, expansion in _RETRIEVAL_EXPANSIONS
                if pattern.search(request_text)
            )
        )
        retrieved = self.store.search(search_text, limit=self.retrieval_limit)
        if not retrieved:
            issue = VerificationIssue(
                code="RETRIEVAL_EMPTY",
                message="No approved biblical evidence was retrieved; generation is withheld.",
                decision=PipelineDecision.CORRECT,
                severity=IssueSeverity.CRITICAL,
                field_path="evidence",
            )
            report = VerificationReport(
                decision=PipelineDecision.CORRECT,
                issues=(issue,),
                checks={"retrieval_present": False},
            )
            return AgentResult(
                report.decision,
                report,
                None,
                _safe_delivery(report),
                0,
                (),
            )
        retrieved = self.store.expand_with_neighbors(
            retrieved,
            additional_limit=self.graph_expansion_limit,
        )

        corpora = self.store.export_corpora(retrieved)
        pipeline = InferenceReviewPipeline(
            commandment_policy=self.commandment_policy,
            content_reviewer=self.content_reviewer,
            citation_verifier=CitationVerifier(
                corpora,
                approved_source_ids=set(corpora),
            ),
            organizational_source_ids=self.store.organizational_source_ids(),
        )
        prompt = self._request_prompt(request_text, retrieved)
        answer: MoralAnswer | None = None
        report: VerificationReport | None = None

        for attempt in range(1, self.max_corrections + 2):
            raw = self.backend.generate(system=self.system_prompt, user=prompt)
            try:
                answer = MoralAnswer.from_dict(_extract_json_object(raw))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                issue = VerificationIssue(
                    code="MODEL_OUTPUT_SCHEMA_INVALID",
                    message=str(exc),
                    decision=PipelineDecision.CORRECT,
                    severity=IssueSeverity.CRITICAL,
                    field_path="model_output",
                )
                report = VerificationReport(
                    decision=PipelineDecision.CORRECT,
                    issues=(issue,),
                    checks={"schema_valid": False},
                )
            else:
                report = pipeline.review(answer)
                if answer.request_text != request_text:
                    mismatch = VerificationIssue(
                        code="REQUEST_TEXT_MISMATCH",
                        message="The structured answer does not preserve the user's exact request text.",
                        decision=PipelineDecision.CORRECT,
                        field_path="request_text",
                    )
                    combined = (*report.issues, mismatch)
                    report = VerificationReport(
                        decision=strongest_decision(combined),
                        issues=combined,
                        checks={**report.checks, "request_preserved": False},
                    )

            if report.decision is PipelineDecision.RELEASE and answer is not None:
                return AgentResult(
                    report.decision,
                    report,
                    answer,
                    render_moral_answer(answer),
                    attempt,
                    tuple(retrieved),
                )
            if report.decision in {PipelineDecision.REFUSE, PipelineDecision.ESCALATE}:
                return AgentResult(
                    report.decision,
                    report,
                    None,
                    _safe_delivery(report),
                    attempt,
                    tuple(retrieved),
                )
            if attempt <= self.max_corrections:
                prompt = self._correction_prompt(request_text, retrieved, raw, report)

        assert report is not None
        return AgentResult(
            report.decision,
            report,
            None,
            _safe_delivery(report),
            self.max_corrections + 1,
            tuple(retrieved),
        )

    @staticmethod
    def _request_prompt(request_text: str, retrieved: list[Passage]) -> str:
        evidence = [
            {
                "source_id": item.source_id,
                "reference": item.reference,
                "language": item.language,
                "exact_text": item.text,
                "context": item.context,
            }
            for item in retrieved
        ]
        return (
            "Return exactly one JSON object matching schemas/moral-answer.schema.json. "
            "Use only the retrieved evidence below for quotations and references. Do not reveal hidden chain-of-thought. "
            "Assess commandments 5-10 explicitly and commandments 1-4 when relevant.\n\n"
            f"REQUEST:\n{request_text}\n\nRETRIEVED_EVIDENCE:\n{json.dumps(evidence, ensure_ascii=False, indent=2)}"
        )

    def _correction_prompt(
        self,
        request_text: str,
        retrieved: list[Passage],
        raw: str,
        report: VerificationReport,
    ) -> str:
        issues = [{"code": item.code, "message": item.message} for item in report.issues]
        return (
            self._request_prompt(request_text, retrieved)
            + "\n\nThe previous candidate failed deterministic verification. Correct every issue and return a full replacement JSON object."
            + f"\nISSUES:\n{json.dumps(issues, indent=2)}"
            + f"\nPREVIOUS_CANDIDATE:\n{raw}"
        )


def _extract_json_object(raw: str) -> dict[str, object]:
    value = raw.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end < start:
        raise json.JSONDecodeError("model output contains no JSON object", value, 0)
    parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("model output JSON root is not an object", value, start)
    return parsed


def _safe_delivery(report: VerificationReport) -> str:
    if report.decision is PipelineDecision.ESCALATE:
        return (
            "This may involve imminent danger or another high-impact risk that I cannot resolve "
            "autonomously. Move to safety and contact local emergency services or a qualified human "
            "now. In the United States, call 911 for immediate danger or call/text 988 for a suicide "
            "or mental-health crisis."
        )
    if report.decision is PipelineDecision.REFUSE:
        return (
            "I cannot help carry out that request because it conflicts with the system's safety and "
            "commandment rules. I can help identify truthful, lawful options that protect life, "
            "property, fidelity, and vulnerable people."
        )
    return (
        "I cannot provide a verified answer because the approved evidence or validation requirements "
        "were not met. No unverified model output has been released."
    )
