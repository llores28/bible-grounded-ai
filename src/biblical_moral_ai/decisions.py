"""Shared decision aggregation for independent verification stages."""

from __future__ import annotations

from collections.abc import Iterable

from .schemas import PipelineDecision, VerificationIssue

_DECISION_PRIORITY = {
    PipelineDecision.RELEASE: 0,
    PipelineDecision.CORRECT: 1,
    PipelineDecision.ESCALATE: 2,
    PipelineDecision.REFUSE: 3,
}


def strongest_decision(issues: Iterable[VerificationIssue]) -> PipelineDecision:
    decision = PipelineDecision.RELEASE
    for issue in issues:
        if _DECISION_PRIORITY[issue.decision] > _DECISION_PRIORITY[decision]:
            decision = issue.decision
    return decision
