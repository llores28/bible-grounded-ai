# Advanced Content Review Workflow

The advanced content reviewer is a fail-closed quality layer for generated answers and governed repository content. It detects contradictions and imprecision; it does not claim that a heuristic can prove every theological statement true or silently rewrite disputed doctrine.

## Runtime answer review

Every `MoralAnswer` passes through `AdvancedContentReviewer` before citation and safety release. The reviewer checks:

- opposite versions of the same proposition across facts, conclusions, alternatives, and practical options;
- configured contradiction families involving certainty, evidence status, prophetic interpretation, AI authority, and forgiveness or access;
- high confidence paired with missing information or pending evidence review;
- Hebrew, Aramaic, or Koine Greek conclusions without reviewed source-language support;
- disagreement between a commandment verdict and its rationale;
- unresolved authoring markers, vague language, duplicated items, and excessively long action sentences.

Every issue includes an exact code, field path, and suggested correction. A contradiction or precision defect returns `correct` or `escalate`; the answer is not delivered. The full policy, citation, content, and pastoral-safety pipeline must pass again after revision.

## Repository audit

Run:

```powershell
$env:PYTHONPATH='src'
python -m biblical_moral_ai audit-content
```

The repository audit checks current machine-readable invariants rather than trusting prose counters. It compares the canon size, pilot queue and candidate counts, deception coverage and catch-all, approved source roles, selected documentation claims, local Markdown links, and unresolved markers. A stale count or broken governed link blocks the audit and includes a file path and suggested correction.

## Limits and human review

Automated contradiction checks are deliberately conservative. They can detect explicit inconsistency, unsupported certainty, stale facts, and unclear language, but they cannot settle every interpretive dispute or infer an author's hidden intent. Scripture quotations, source-language conclusions, disputed doctrine, prophecy, abuse, violence, force, and other high-impact judgments retain their existing evidence and independent-review requirements.

The reviewer never auto-edits Scripture, quotations, evidence records, reviewer decisions, doctrine, or safety conclusions. Safe mechanical correction may be proposed, but the revised content must be revalidated and substantive changes must receive the required human review.
