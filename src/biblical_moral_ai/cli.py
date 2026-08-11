"""Command-line interface for validation, policy review, and release gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .arithmetic import calculate, verify_equation
from .citation import CitationVerifier
from .corpus_build import build_approved_evidence
from .evidence_store import EvidenceStore
from .hardware import inspect_cuda
from .inference import BiblicalMoralAgent, LocalChatBackend
from .pilot import PilotWorkflow
from .pilot_authoring import PilotDraftWorkflow
from .pilot_candidates import PilotCandidateWorkflow
from .pilot_seed import PilotSeedBuilder
from .pipeline import InferenceReviewPipeline
from .policy import CommandmentPolicyEngine
from .preflight import ProjectPreflight
from .registry import load_commandment_rules, load_json
from .release import ReleaseGateEvaluator, ReleaseMetrics
from .review_ledger import ReviewLedgerValidator
from .schemas import MoralAnswer
from .training import TrainingBlockedError, inspect_training_request, run_training


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, default=str))


def _pipeline(root: Path, corpus_path: Path) -> InferenceReviewPipeline:
    corpus_payload = load_json(corpus_path)
    corpora = corpus_payload.get("sources", corpus_payload)
    return InferenceReviewPipeline(
        commandment_policy=CommandmentPolicyEngine(
            load_commandment_rules(root / "configs/commandments.json")
        ),
        citation_verifier=CitationVerifier(corpora),
        organizational_source_ids=corpus_payload.get("organizational_source_ids", []),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="biblical-moral-ai")
    parser.add_argument("--root", default=".", help="project root")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="validate repository structure and registries")
    commands.add_parser("preflight", help="check whether full training is allowed")
    commands.add_parser("pilot-preflight", help="validate the 50/20/25 reviewed pilot")
    commands.add_parser(
        "materialize-pilot", help="materialize training rows after pilot preflight passes"
    )
    commands.add_parser(
        "audit-pilot-drafts", help="verify the 50/20/25 draft queue against pinned evidence"
    )
    commands.add_parser(
        "build-authoring-packets", help="snapshot exact evidence for pilot authors"
    )
    commands.add_parser(
        "audit-pilot-candidates", help="validate fully authored candidates before review"
    )
    seed = commands.add_parser(
        "seed-pilot-candidates",
        help="create deterministic AI-authored candidates for human revision and review",
    )
    seed.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing candidate files deliberately",
    )
    commands.add_parser(
        "build-candidate-review-packets",
        help="bind validated candidate records to exact evidence for review",
    )
    commands.add_parser(
        "assign-pilot-reviewers", help="assign qualified reviewers to validated candidates"
    )
    commands.add_parser(
        "validate-review-ledger", help="validate blinded reviews and adjudications"
    )
    commands.add_parser(
        "finalize-reviewed-pilot", help="write accepted pilot splits after unanimous review"
    )
    commands.add_parser(
        "write-pilot-audit-receipt", help="write a hash-bound CPU validation receipt"
    )
    build = commands.add_parser(
        "build-evidence", help="fetch, verify, and atomically build all approved evidence"
    )
    build.add_argument("--cache", default="data/cache/upstream")
    build.add_argument("--artifacts", default="data/artifacts/canonical")
    build.add_argument("--database", default="data/index/biblical_evidence.sqlite3")
    build.add_argument("--corpus", default="data/index/citation_corpus.json")
    build.add_argument("--manifest", default="data/index/evidence_manifest.json")
    build.add_argument("--fetch", action="store_true")
    cuda = commands.add_parser("cuda-check", help="inspect local CUDA readiness")
    cuda.add_argument("--minimum-vram-gib", type=float, default=24.0)
    arithmetic = commands.add_parser("calculate", help="run safe decimal arithmetic")
    arithmetic.add_argument("expression")
    arithmetic.add_argument("--unit", default="")
    arithmetic.add_argument("--equation", action="store_true")
    review = commands.add_parser("review-answer", help="verify a structured answer JSON")
    review.add_argument("answer")
    review.add_argument("--corpus", required=True)
    release = commands.add_parser("check-release", help="evaluate release metrics")
    release.add_argument("metrics")
    ingest = commands.add_parser(
        "ingest-corpus", help="ingest one approved, digest-matched corpus artifact"
    )
    ingest.add_argument("artifact")
    ingest.add_argument("--database", default="data/index/biblical_evidence.sqlite3")
    ingest.add_argument("--registry", default="configs/data/source_registry.json")
    ingest.add_argument("--canon", default="configs/canon.json")
    search = commands.add_parser("search-evidence", help="search approved biblical evidence")
    search.add_argument("query")
    search.add_argument("--database", default="data/index/biblical_evidence.sqlite3")
    search.add_argument("--limit", type=int, default=12)
    lexicon = commands.add_parser("search-lexicon", help="search auxiliary language dictionaries")
    lexicon.add_argument("query")
    lexicon.add_argument("--database", default="data/index/biblical_evidence.sqlite3")
    lexicon.add_argument("--language")
    lexicon.add_argument("--limit", type=int, default=12)
    answer = commands.add_parser(
        "answer", help="run retrieval-first inference against a local endpoint"
    )
    answer.add_argument("request")
    answer.add_argument("--database", default="data/index/biblical_evidence.sqlite3")
    answer.add_argument("--config", default="configs/inference/local_vllm.json")
    training = commands.add_parser("train", help="inspect or execute a configured training stage")
    training.add_argument("config")
    training.add_argument("--execute", action="store_true", help="execute after all gates pass")
    training.add_argument("--smoke-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()

    if args.command == "validate":
        report = ProjectPreflight(root).validate_structure()
        _json(report.to_dict())
        return 0 if report.ready else 1
    if args.command == "preflight":
        report = ProjectPreflight(root).training_readiness()
        _json(report.to_dict())
        return 0 if report.ready else 2
    if args.command == "pilot-preflight":
        report = PilotWorkflow(root).readiness()
        _json(report.to_dict())
        return 0 if report.ready else 2
    if args.command == "materialize-pilot":
        try:
            outputs = PilotWorkflow(root).materialize()
        except ValueError as exc:
            _json({"status": "blocked", "reason": str(exc)})
            return 2
        _json({"status": "materialized", "outputs": outputs})
        return 0
    if args.command == "audit-pilot-drafts":
        report = PilotDraftWorkflow(root).audit()
        _json(report.to_dict())
        return 0 if report.passed else 2
    if args.command == "build-authoring-packets":
        try:
            result = PilotDraftWorkflow(root).build_authoring_packets()
        except ValueError as exc:
            _json({"status": "blocked", "reason": str(exc)})
            return 2
        _json(result)
        return 0
    if args.command == "audit-pilot-candidates":
        report = PilotCandidateWorkflow(root).audit()
        _json(report.to_dict())
        return 0 if report.passed else 2
    if args.command == "seed-pilot-candidates":
        try:
            result = PilotSeedBuilder(root).write_candidates(overwrite=args.overwrite)
        except ValueError as exc:
            _json({"status": "blocked", "reason": str(exc)})
            return 2
        _json(result)
        return 0
    if args.command == "build-candidate-review-packets":
        try:
            result = PilotCandidateWorkflow(root).build_review_packets()
        except ValueError as exc:
            _json({"status": "blocked", "reason": str(exc)})
            return 2
        _json(result)
        return 0
    if args.command == "assign-pilot-reviewers":
        try:
            result = PilotCandidateWorkflow(root).assign_reviewers()
        except ValueError as exc:
            _json({"status": "blocked", "reason": str(exc)})
            return 2
        _json(result)
        return 0
    if args.command == "validate-review-ledger":
        report = ReviewLedgerValidator(root).validate()
        _json(report.to_dict())
        return 0 if report.passed else 2
    if args.command == "finalize-reviewed-pilot":
        try:
            result = PilotCandidateWorkflow(root).finalize_reviewed_pilot()
        except ValueError as exc:
            _json({"status": "blocked", "reason": str(exc)})
            return 2
        _json(result)
        return 0
    if args.command == "write-pilot-audit-receipt":
        result = PilotDraftWorkflow(root).write_cpu_audit_receipt()
        _json(result)
        return 0 if result["status"] == "ready" else 2
    if args.command == "build-evidence":
        result = build_approved_evidence(
            root=root,
            cache_dir=root / args.cache,
            artifact_dir=root / args.artifacts,
            database_path=root / args.database,
            citation_corpus_path=root / args.corpus,
            manifest_path=root / args.manifest,
            fetch=args.fetch,
        )
        _json(result.to_dict())
        return 0
    if args.command == "cuda-check":
        report = inspect_cuda(minimum_vram_gib=args.minimum_vram_gib)
        _json(report.to_dict())
        return 0 if report.ready else 2
    if args.command == "calculate":
        if args.equation:
            _json({"equation": args.expression, "valid": verify_equation(args.expression)})
        else:
            result = calculate(args.expression, args.unit)
            _json(
                {"expression": result.expression, "value": str(result.value), "unit": result.unit}
            )
        return 0
    if args.command == "review-answer":
        answer = MoralAnswer.from_dict(load_json(args.answer))
        report = _pipeline(root, Path(args.corpus)).review(answer)
        _json(report.to_dict())
        return 0 if report.passed else 2
    if args.command == "check-release":
        metrics = ReleaseMetrics.from_dict(load_json(args.metrics))
        result = ReleaseGateEvaluator().evaluate(metrics)
        _json(result.to_dict())
        return 0 if result.approved else 2
    if args.command == "ingest-corpus":
        database = root / args.database
        database.parent.mkdir(parents=True, exist_ok=True)
        with EvidenceStore(database) as store:
            store.import_artifact(root / args.artifact, root / args.registry, root / args.canon)
        _json({"status": "ingested", "artifact": args.artifact, "database": str(database)})
        return 0
    if args.command == "search-evidence":
        with EvidenceStore(root / args.database) as store:
            passages = store.search(args.query, limit=args.limit)
        _json(
            [
                {
                    "source_id": item.source_id,
                    "reference": item.reference,
                    "language": item.language,
                    "text": item.text,
                    "context": item.context,
                }
                for item in passages
            ]
        )
        return 0
    if args.command == "search-lexicon":
        with EvidenceStore(root / args.database) as store:
            entries = store.search_lexicon(
                args.query,
                language=args.language,
                limit=args.limit,
            )
        _json(
            [
                {
                    "source_id": item.source_id,
                    "entry_id": item.entry_id,
                    "language": item.language,
                    "lemma": item.lemma,
                    "transliteration": item.transliteration,
                    "gloss": item.gloss,
                    "definition": item.definition,
                    "source_ref": item.source_ref,
                }
                for item in entries
            ]
        )
        return 0
    if args.command == "answer":
        config = load_json(root / args.config)
        backend = LocalChatBackend(
            endpoint=config["endpoint"],
            model=config["model"],
            allow_remote=bool(config.get("allow_remote", False)),
        )
        with EvidenceStore(root / args.database) as store:
            result = BiblicalMoralAgent(
                root=root,
                store=store,
                backend=backend,
                retrieval_limit=int(config.get("retrieval_limit", 12)),
                graph_expansion_limit=int(config.get("graph_expansion_limit", 8)),
                max_corrections=int(config.get("max_corrections", 1)),
            ).answer(args.request)
        payload = {
            "decision": result.decision.value,
            "attempts": result.attempts,
            "retrieved": [f"{item.source_id} {item.reference}" for item in result.retrieved],
            "issues": [
                {"code": item.code, "message": item.message} for item in result.report.issues
            ],
            "answer": result.delivery_text,
        }
        _json(payload)
        return 0 if result.decision.value == "release" else 2
    if args.command == "train":
        if not args.execute:
            _json(inspect_training_request(args.config, root=root))
            return 0
        try:
            manifest = run_training(args.config, root=root, smoke_test=args.smoke_test)
        except TrainingBlockedError as exc:
            _json({"status": "blocked", "reason": str(exc)})
            return 2
        _json({"status": "completed", "manifest": str(manifest)})
        return 0
    return 1
